//go:build ignore

// Command gen produces encode_amd64.s with go-asmgen: a vectorised base32
// (RFC 4648 StdEncoding) encoder, both an SSE2/SSSE3 path (5 input bytes -> 8
// chars per iteration) and an AVX2 path (10 -> 16 per 256-bit block, two groups
// processed one per 128-bit lane via VINSERTI128).
//
// Algorithm (per 5-byte group -> 8 chars): a PSHUFB spreads the input into eight
// 16-bit lanes, each lane holding the (big-endian) 16-bit window that contains
// one output char's 5 bits. A single per-lane unsigned high-multiply (PMULHUW)
// by lane-specific powers of two shifts each char's 5-bit field down to bits
// [4:0]; PAND 0x1f isolates it (value 0..31). A second PSHUFB packs the eight
// values (which sit in the low byte of each lane) into the low 8 bytes, then a
// two-range ASCII map turns value v into its char: v<26 -> 'A'+v, v>=26 ->
// '2'+(v-26). The map is built as v + 65 - (PCMPGTB(v,25) & 41); the result's
// low 8 bytes are stored with MOVQ. Constant tables come from emit.File.Data.
// Run: go run encode_gen.go
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

// fieldInfo returns, for output char i, the input byte holding the high part of
// its 5-bit field, the byte holding the low part, and the bit offset p of the
// field's LSB within the big-endian 16-bit window [hiByte:loByte]. The PMULHUW
// multiplier for that lane is 2^(16-p).
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

// spreadConstants builds the PSHUFB spread mask (16 bytes) and the per-lane
// PMULHUW multiplier table (16 bytes = eight 16-bit values).
func spreadConstants() (shuf, mul []byte) {
	shuf = make([]byte, 16)
	mul = make([]byte, 16)
	for i := 0; i < 8; i++ {
		hb, lb, p := fieldInfo(i)
		shuf[2*i] = byte(lb)   // low byte of lane <- loByte
		shuf[2*i+1] = byte(hb) // high byte of lane <- hiByte
		m := uint16(1) << uint(16-p)
		mul[2*i] = byte(m)
		mul[2*i+1] = byte(m >> 8)
	}
	return
}

// packMask gathers the eight value bytes (low byte of each 16-bit lane, i.e.
// byte positions 0,2,4,...,14) into the low 8 bytes; high 8 bytes are zeroed
// (index 0x80 -> 0 in PSHUFB) so they don't corrupt the later compare.
func packMask() []byte {
	m := make([]byte, 16)
	for i := 0; i < 8; i++ {
		m[i] = byte(2 * i)
	}
	for i := 8; i < 16; i++ {
		m[i] = 0x80
	}
	return m
}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		nil,
	)
}

func main() {
	f := emit.NewFile("amd64")

	shufBytes, mulBytes := spreadConstants()
	packBytes := packMask()

	// ---- SSE2/SSSE3: 5 -> 8 ----
	shuf := f.Data("shuf", shufBytes)
	mul := f.Data("mul", mulBytes)
	mask1f := f.Data("mask1f", repByte(0x1f, 16))
	pack := f.Data("pack", packBytes)
	c25 := f.Data("c25", repByte(25, 16))
	c65 := f.Data("c65", repByte(65, 16))
	c41 := f.Data("c41", repByte(41, 16))

	s := amd64.NewFunc("encodeBlocksSSE", sig(), 0)
	s.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("n", "CX").
		Raw("MOVOU %s+0(SB), X8", shuf).
		Raw("MOVOU %s+0(SB), X9", mul).
		Raw("MOVOU %s+0(SB), X10", pack).
		Raw("MOVOU %s+0(SB), X11", c25).
		Raw("MOVOU %s+0(SB), X12", c65).
		Raw("MOVOU %s+0(SB), X13", c41).
		Raw("TESTQ CX, CX").Raw("JZ sdone").
		Label("sloop").
		Raw("MOVOU (SI), X0").Raw("PSHUFB X8, X0"). // spread -> 8x 16-bit windows
		Raw("PMULHUW X9, X0").                      // shift each field to bits[4:0]
		Raw("PAND %s+0(SB), X0", mask1f).           // & 0x1f
		Raw("PSHUFB X10, X0").                      // pack 8 values into low 8 bytes
		Raw("MOVO X0, X1").Raw("PCMPGTB X11, X1").  // mask = v>25 ? 0xff : 0
		Raw("PAND X13, X1").                        // mask & 41
		Raw("PADDB X12, X0").                       // v + 65
		Raw("PSUBB X1, X0").                        // - (mask & 41)
		Raw("MOVQ X0, (DI)").                       // store low 8 bytes
		Raw("ADDQ $5, SI").Raw("ADDQ $8, DI").Raw("DECQ CX").Raw("JNZ sloop").
		Label("sdone").Ret()
	f.Add(s.Func())

	// ---- AVX2: 10 -> 16, two 5-byte groups, one per 128-bit lane ----
	// lane0 = src[0..15], lane1 = src[5..20] (VINSERTI128). The same 16-byte
	// spread/pack/multiply masks, replicated to 256 bits, then process both lanes
	// in parallel; the two packed 8-byte results land in bytes [0:8] of lane0 and
	// [0:8] of lane1, i.e. dst[0:8] and dst[8:16] after a VPERMQ gather.
	shufb := f.Data("shufb", rep(shufBytes, 2))
	mulb := f.Data("mulb", rep(mulBytes, 2))
	maskb := f.Data("maskb", rep(repByte(0x1f, 16), 2))
	packb := f.Data("packb", rep(packBytes, 2))
	c25b := f.Data("c25b", repByte(25, 32))
	c65b := f.Data("c65b", repByte(65, 32))
	c41b := f.Data("c41b", repByte(41, 32))

	vv := amd64.NewFunc("encodeBlocksAVX2", sig(), 0)
	vv.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("n", "CX").
		Raw("VMOVDQU %s+0(SB), Y8", shufb).
		Raw("VMOVDQU %s+0(SB), Y9", mulb).
		Raw("VMOVDQU %s+0(SB), Y10", maskb).
		Raw("VMOVDQU %s+0(SB), Y11", packb).
		Raw("VMOVDQU %s+0(SB), Y12", c25b).
		Raw("VMOVDQU %s+0(SB), Y13", c65b).
		Raw("VMOVDQU %s+0(SB), Y14", c41b).
		Raw("TESTQ CX, CX").Raw("JZ vdone").
		// 2x-unrolled steady state: two independent blocks (Y0/Y1 and Y2/Y3)
		// interleaved to hide the long PSHUFB->PMULHUW->...->VPERMQ dependency
		// chain. Block A reads src[0..20], block B reads src[10..30].
		Label("vpair").
		Raw("CMPQ CX, $2").Raw("JLT vsingle").
		Raw("VMOVDQU (SI), X0").Raw("VINSERTI128 $1, 5(SI), Y0, Y0").
		Raw("VMOVDQU 10(SI), X2").Raw("VINSERTI128 $1, 15(SI), Y2, Y2").
		Raw("VPSHUFB Y8, Y0, Y0").Raw("VPSHUFB Y8, Y2, Y2").
		Raw("VPMULHUW Y9, Y0, Y0").Raw("VPMULHUW Y9, Y2, Y2").
		Raw("VPAND Y10, Y0, Y0").Raw("VPAND Y10, Y2, Y2").
		Raw("VPSHUFB Y11, Y0, Y0").Raw("VPSHUFB Y11, Y2, Y2").
		Raw("VPCMPGTB Y12, Y0, Y1").Raw("VPCMPGTB Y12, Y2, Y3").
		Raw("VPAND Y14, Y1, Y1").Raw("VPAND Y14, Y3, Y3").
		Raw("VPADDB Y13, Y0, Y0").Raw("VPADDB Y13, Y2, Y2").
		Raw("VPSUBB Y1, Y0, Y0").Raw("VPSUBB Y3, Y2, Y2").
		Raw("VPERMQ $0x08, Y0, Y0").Raw("VPERMQ $0x08, Y2, Y2").
		Raw("VMOVDQU X0, (DI)").Raw("VMOVDQU X2, 16(DI)").
		Raw("ADDQ $20, SI").Raw("ADDQ $32, DI").Raw("SUBQ $2, CX").Raw("JMP vpair").
		Label("vsingle").
		Raw("TESTQ CX, CX").Raw("JZ vdone").
		Raw("VMOVDQU (SI), X0").              // lane0 <- src[0..15]
		Raw("VINSERTI128 $1, 5(SI), Y0, Y0"). // lane1 <- src[5..20]
		Raw("VPSHUFB Y8, Y0, Y0").            // spread (per 128-bit lane)
		Raw("VPMULHUW Y9, Y0, Y0").           // shift fields to bits[4:0]
		Raw("VPAND Y10, Y0, Y0").             // & 0x1f
		Raw("VPSHUFB Y11, Y0, Y0").           // pack each lane to its low 8 bytes
		Raw("VPCMPGTB Y12, Y0, Y1").          // mask = v>25
		Raw("VPAND Y14, Y1, Y1").             // mask & 41
		Raw("VPADDB Y13, Y0, Y0").            // v + 65
		Raw("VPSUBB Y1, Y0, Y0").             // - (mask & 41)
		Raw("VPERMQ $0x08, Y0, Y0").          // gather lane0[0:8],lane1[0:8] -> [0:16]
		Raw("VMOVDQU X0, (DI)").              // store 16 chars
		Label("vdone").Raw("VZEROUPPER").Ret()
	f.Add(vv.Func())

	if err := os.WriteFile("encode_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_amd64.s")
}
