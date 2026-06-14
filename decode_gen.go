//go:build ignore

// Command gen produces decode_amd64.s with go-asmgen: a vectorised base32
// (RFC 4648 StdEncoding) decoder, an SSE2/SSSE3 path (8 input chars -> 5 bytes
// per iteration) and an AVX2 path (two blocks per 256-bit register, one per
// 128-bit lane).
//
// The decoder is the inverse of the encoder. Per 8-char block -> 5 bytes:
//
//  1. Validate + map ASCII -> 5-bit value, per lane, with two range checks done
//     via saturating unsigned subtract (PCMPGTB is signed, so the classic
//     PSUBUSB-then-compare-zero trick is used):
//       az = (PSUBUSB(c-'A', 25) == 0)   -> c in 'A'..'Z'  (values 0..25)
//       tw = (PSUBUSB(c-'2',  5) == 0)   -> c in '2'..'7'  (values 26..31)
//       v  = ((c-65) & az) | ((c-24) & tw)
//     A block is decoded only if all 8 lanes are valid; the function counts
//     consumed blocks and stops before the first block containing any invalid
//     char (the caller hands the remainder, plus the always-reserved final
//     block, to encoding/base32 so padding/tail/error offsets stay identical).
//  2. Spread each value into the low byte of its 16-bit lane (the value already
//     sits there), then PMULLW (multiply *low* word) by the per-lane 2^p shifts
//     it left into its big-endian 16-bit output window — the exact inverse of the
//     encoder's PMULHUW-by-2^(16-p) right shift.
//  3. Three PSHUFB gathers + OR scatter each window's high/low byte into the five
//     output bytes (each output byte takes up to three overlapping contributions;
//     single-byte windows contribute only their high byte, whose low byte is
//     structurally zero, so two passes of the third gather slot are vacant).
//  4. MOVL+MOVB (or a wide store with reserved slack) writes the 5 bytes.
//
// Constant tables come from emit.File.Data. Run: go run decode_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func rep(v []byte, times int) []byte {
	var b []byte
	for i := 0; i < times; i++ {
		b = append(b, v...)
	}
	return b
}
func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

// fieldInfo returns, for value i, the output byte holding the high part of its
// 5-bit field, the byte holding the low part, and the bit offset p of the field's
// LSB within the big-endian 16-bit window [hiByte:loByte]. The PMULLW multiplier
// for that lane is 2^p (a left shift).
func fieldInfo(i int) (hiByte, loByte, p int) {
	topVal := 39 - 5*i
	botVal := 35 - 5*i
	hiByte = (39 - topVal) / 8
	loByte = (39 - botVal) / 8
	base := 32 - 8*loByte
	off := botVal - base
	if loByte == hiByte {
		p = 8 + off
	} else {
		p = off
	}
	return
}

// spreadMask scatters the eight packed value bytes (positions 0..7) into the low
// byte of each 16-bit lane (positions 0,2,4,...,14); the high byte of each lane
// is zeroed (0x80 -> 0 in PSHUFB) so PMULLW sees value 0..31 per word.
func spreadMask() []byte {
	m := make([]byte, 16)
	for x := range m {
		m[x] = 0x80
	}
	for i := 0; i < 8; i++ {
		m[2*i] = byte(i)
	}
	return m
}

// mulTable builds the per-lane PMULLW multiplier (16 bytes = eight 16-bit values
// 2^p, little-endian per lane).
func mulTable() []byte {
	mul := make([]byte, 16)
	for i := 0; i < 8; i++ {
		_, _, p := fieldInfo(i)
		m := uint16(1) << uint(p)
		mul[2*i] = byte(m)
		mul[2*i+1] = byte(m >> 8)
	}
	return mul
}

// gatherMasks builds the three PSHUFB gather controls. After PMULLW, lane i sits
// at bytes [2i (low), 2i+1 (high)]. Output byte j collects: each value's high
// byte (vec[2i+1]) goes to byte hiByte(i); for two-byte windows (loByte!=hiByte)
// the low byte (vec[2i]) goes to byte loByte(i). 0x80 -> zero in PSHUFB.
func gatherMasks() (g0, g1, g2 []byte) {
	src := make([][]int, 5)
	for i := 0; i < 8; i++ {
		hb, lb, _ := fieldInfo(i)
		src[hb] = append(src[hb], 2*i+1) // high byte of window
		if lb != hb {
			src[lb] = append(src[lb], 2*i) // low byte of window
		}
	}
	g := [3][]byte{}
	for k := range g {
		g[k] = make([]byte, 16)
		for x := range g[k] {
			g[k][x] = 0x80
		}
	}
	for j := 0; j < 5; j++ {
		for k, s := range src[j] {
			g[k][j] = byte(s)
		}
	}
	return g[0], g[1], g[2]
}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Int64)},
	)
}

func main() {
	f := emit.NewFile("amd64")

	mulBytes := mulTable()
	spreadBytes := spreadMask()
	g0b, g1b, g2b := gatherMasks()

	// ---- SSE2/SSSE3: 8 chars -> 5 bytes ----
	cA := f.Data("cA", repByte('A', 16))
	c25 := f.Data("c25", repByte(25, 16))
	c2 := f.Data("c2", repByte('2', 16))
	c5 := f.Data("c5", repByte(5, 16))
	c65 := f.Data("c65", repByte(65, 16))
	c24 := f.Data("c24", repByte(24, 16))
	spread := f.Data("spread", spreadBytes)
	mul := f.Data("mul", mulBytes)
	g0 := f.Data("g0", g0b)
	g1 := f.Data("g1", g1b)
	g2 := f.Data("g2", g2b)

	s := amd64.NewFunc("decodeBlocksSSE", sig(), 0)
	s.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("n", "CX").
		Raw("XORQ AX, AX").  // AX = blocks decoded
		Raw("TESTQ CX, CX").Raw("JZ ddone").
		Label("dloop").
		Raw("MOVQ (SI), X0"). // load 8 chars into low 8 bytes
		// validate: az = (subusb(c-'A',25)==0), tw = (subusb(c-'2',5)==0)
		Raw("MOVO X0, X1").Raw("PSUBB %s+0(SB), X1", cA). // X1 = c-'A'
		Raw("PSUBUSB %s+0(SB), X1", c25).                 // saturating - 25
		Raw("PXOR X4, X4").Raw("PCMPEQB X4, X1").         // X1 = az mask (0xff where valid A..Z)
		Raw("MOVO X0, X2").Raw("PSUBB %s+0(SB), X2", c2). // X2 = c-'2'
		Raw("PSUBUSB %s+0(SB), X2", c5).                  // saturating - 5
		Raw("PXOR X5, X5").Raw("PCMPEQB X5, X2").         // X2 = tw mask
		// block validity: every one of the 8 low lanes must be az|tw.
		Raw("MOVO X1, X3").Raw("POR X2, X3"). // X3 = valid mask per lane
		Raw("PMOVMSKB X3, DX").Raw("ANDQ $0xff, DX").
		Raw("CMPQ DX, $0xff").Raw("JNE ddone"). // any invalid lane -> stop
		// value = ((c-65)&az) | ((c-24)&tw)
		Raw("MOVO X0, X6").Raw("PSUBB %s+0(SB), X6", c65).Raw("PAND X1, X6"). // (c-65)&az
		Raw("PSUBB %s+0(SB), X0", c24).Raw("PAND X2, X0").                    // (c-24)&tw
		Raw("POR X6, X0").               // X0 = 8 values packed in bytes 0..7
		Raw("PSHUFB %s+0(SB), X0", spread). // spread to low byte of each 16-bit lane
		Raw("PMULLW %s+0(SB), X0", mul).    // value << p per 16-bit lane
		// scatter to 5 output bytes via 3 gathers + OR
		Raw("MOVO X0, X7").Raw("PSHUFB %s+0(SB), X7", g0).
		Raw("MOVO X0, X8").Raw("PSHUFB %s+0(SB), X8", g1).Raw("POR X8, X7").
		Raw("PSHUFB %s+0(SB), X0", g2).Raw("POR X0, X7").
		// store 5 bytes: low 4 with MOVL, 5th with PEXTRB
		Raw("MOVD X7, (DI)").
		Raw("PEXTRB $4, X7, 4(DI)").
		Raw("ADDQ $8, SI").Raw("ADDQ $5, DI").Raw("INCQ AX").
		Raw("DECQ CX").Raw("JNZ dloop").
		Label("ddone").Raw("MOVQ AX, ret+56(FP)").Ret()
	f.Add(s.Func())

	// ---- AVX2: two blocks, one per 128-bit lane ----
	cAb := f.Data("cAb", rep(repByte('A', 16), 2))
	c25b := f.Data("c25b", rep(repByte(25, 16), 2))
	c2b := f.Data("c2b", rep(repByte('2', 16), 2))
	c5b := f.Data("c5b", rep(repByte(5, 16), 2))
	c65b := f.Data("c65b", rep(repByte(65, 16), 2))
	c24b := f.Data("c24b", rep(repByte(24, 16), 2))
	spreadb := f.Data("spreadb", rep(spreadBytes, 2))
	mulb := f.Data("mulb", rep(mulBytes, 2))
	g0bb := f.Data("g0bb", rep(g0b, 2))
	g1bb := f.Data("g1bb", rep(g1b, 2))
	g2bb := f.Data("g2bb", rep(g2b, 2))

	vv := amd64.NewFunc("decodeBlocksAVX2", sig(), 0)
	vv.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("n", "CX").
		Raw("XORQ AX, AX").
		Raw("VMOVDQU %s+0(SB), Y10", cAb).
		Raw("VMOVDQU %s+0(SB), Y11", c25b).
		Raw("VMOVDQU %s+0(SB), Y12", c2b).
		Raw("VMOVDQU %s+0(SB), Y13", c5b).
		Raw("VMOVDQU %s+0(SB), Y14", c65b).
		Raw("VMOVDQU %s+0(SB), Y15", c24b).
		Label("vloop").
		Raw("CMPQ CX, $2").Raw("JLT vtail").
		// lane0 holds chars 0..7, lane1 holds chars 8..15, each in its low 8 bytes.
		Raw("VMOVQ (SI), X0").              // X0 bytes[0:8] = chars 0..7
		Raw("VMOVQ 8(SI), X1").             // X1 bytes[0:8] = chars 8..15
		Raw("VINSERTI128 $1, X1, Y0, Y0"). // Y0 = lane0:chars0..7, lane1:chars8..15
		// validate per lane
		Raw("VPSUBB Y10, Y0, Y1").Raw("VPSUBUSB Y11, Y1, Y1").
		Raw("VPXOR Y4, Y4, Y4").Raw("VPCMPEQB Y4, Y1, Y1"). // az
		Raw("VPSUBB Y12, Y0, Y2").Raw("VPSUBUSB Y13, Y2, Y2").
		Raw("VPXOR Y5, Y5, Y5").Raw("VPCMPEQB Y5, Y2, Y2"). // tw
		Raw("VPOR Y2, Y1, Y3").                             // valid mask per lane
		Raw("VPMOVMSKB Y3, DX").
		// both blocks valid iff low 8 bits of each 128-bit lane's mask are all set:
		// lane0 -> bits[0:8], lane1 -> bits[16:24].
		Raw("MOVQ DX, BX").Raw("ANDQ $0xff, BX").Raw("CMPQ BX, $0xff").Raw("JNE vtail").
		Raw("MOVQ DX, BX").Raw("SHRQ $16, BX").Raw("ANDQ $0xff, BX").Raw("CMPQ BX, $0xff").Raw("JNE vtail").
		// map values
		Raw("VPSUBB Y14, Y0, Y6").Raw("VPAND Y1, Y6, Y6").
		Raw("VPSUBB Y15, Y0, Y0").Raw("VPAND Y2, Y0, Y0").
		Raw("VPOR Y6, Y0, Y0").
		Raw("VPSHUFB %s+0(SB), Y0, Y0", spreadb).
		Raw("VPMULLW %s+0(SB), Y0, Y0", mulb).
		Raw("VPSHUFB %s+0(SB), Y0, Y7", g0bb).
		Raw("VPSHUFB %s+0(SB), Y0, Y8", g1bb).Raw("VPOR Y8, Y7, Y7").
		Raw("VPSHUFB %s+0(SB), Y0, Y9", g2bb).Raw("VPOR Y9, Y7, Y7").
		// store: lane0's low 5 bytes -> dst[0:5], lane1's low 5 bytes -> dst[5:10]
		Raw("VEXTRACTI128 $1, Y7, X8"). // X8 = lane1 result
		Raw("VMOVD X7, (DI)").Raw("VPEXTRB $4, X7, 4(DI)").
		Raw("VMOVD X8, 5(DI)").Raw("VPEXTRB $4, X8, 9(DI)").
		Raw("ADDQ $16, SI").Raw("ADDQ $10, DI").Raw("ADDQ $2, AX").
		Raw("SUBQ $2, CX").Raw("JMP vloop").
		Label("vtail").
		// one block left (or stopped): fall back to a single SSE-style block in YMM lane0.
		Raw("TESTQ CX, CX").Raw("JZ vdone").
		Raw("VMOVQ (SI), X0").
		Raw("VPSUBB X10, X0, X1").Raw("VPSUBUSB X11, X1, X1").
		Raw("VPXOR X4, X4, X4").Raw("VPCMPEQB X4, X1, X1").
		Raw("VPSUBB X12, X0, X2").Raw("VPSUBUSB X13, X2, X2").
		Raw("VPXOR X5, X5, X5").Raw("VPCMPEQB X5, X2, X2").
		Raw("VPOR X2, X1, X3").
		Raw("VPMOVMSKB X3, DX").Raw("ANDQ $0xff, DX").Raw("CMPQ DX, $0xff").Raw("JNE vdone").
		Raw("VPSUBB X14, X0, X6").Raw("VPAND X1, X6, X6").
		Raw("VPSUBB X15, X0, X0").Raw("VPAND X2, X0, X0").
		Raw("VPOR X6, X0, X0").
		Raw("VPSHUFB %s+0(SB), X0, X0", spread).
		Raw("VPMULLW %s+0(SB), X0, X0", mul).
		Raw("VPSHUFB %s+0(SB), X0, X7", g0).
		Raw("VPSHUFB %s+0(SB), X0, X8", g1).Raw("VPOR X8, X7, X7").
		Raw("VPSHUFB %s+0(SB), X0, X0", g2).Raw("VPOR X0, X7, X7").
		Raw("VMOVD X7, (DI)").Raw("VPEXTRB $4, X7, 4(DI)").
		Raw("ADDQ $8, SI").Raw("ADDQ $5, DI").Raw("INCQ AX").
		Label("vdone").Raw("VZEROUPPER").Raw("MOVQ AX, ret+56(FP)").Ret()
	f.Add(vv.Func())

	if err := os.WriteFile("decode_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_amd64.s")
}
