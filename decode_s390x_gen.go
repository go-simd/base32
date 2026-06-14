//go:build ignore

// Command gen produces decode_s390x.s with go-asmgen: a vectorised base32
// (RFC 4648 StdEncoding) decoder for z13+ (the vector facility is baseline there,
// so there is no runtime feature dispatch — the kernel always runs).
//
// The decoder is the inverse of the encoder. Per 8-char block -> 5 bytes, a
// faithful port of the amd64 SSE decode path (s390x has a genuine vector integer
// multiply-low, VMLH, so it reproduces the PMULLW left-shift trick directly):
//
//  1. Validate + map ASCII -> 5-bit value, per lane, with two unsigned range
//     checks. az = c in 'A'..'Z' = NOT(c-'A' >u 25); tw = c in '2'..'7' =
//     NOT(c-'2' >u 5), using VCHLB (compare-high logical byte) + VNO. value =
//     VSEL between (c-65) and (c-24); lanes matching neither range are invalid.
//     The block is decoded only if all 8 active lanes are valid (checked by
//     extracting the high doubleword of the per-lane valid mask with VLGVG and
//     requiring all ones). The kernel counts consumed blocks and stops before the
//     first invalid block.
//  2. VPERM (`spread`) scatters the eight value bytes into the low byte of each
//     big-endian 16-bit lane.
//  3. VMLHW (the Go mnemonic for VECTOR MULTIPLY LOW, halfword elements) by the
//     per-lane 2^p left-shifts each value into its big-endian 16-bit output
//     window — the exact inverse of the encoder's VMLHH (multiply-high logical)
//     right shift.
//  4. Three VPERM gathers + VO scatter each window's high/low byte into the five
//     output bytes (each takes up to three overlapping contributions).
//  5. VST stores the 16-byte vector; the caller keeps the low 5 bytes.
//
// BIG-ENDIAN: VL puts the lowest memory address into element 0; the VPERM control
// vectors and multiplier table are laid out in big-endian lane order (lane 0 =
// lowest address), matching the amd64 byte layout because amd64's 16-bit windows
// are themselves big-endian. The cross-lane shuffle is pinned by a
// position-dependent qemu fuzz test.
//
// Run: go run decode_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

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

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

// constants builds, in big-endian lane order: the VPERM spread control (value
// byte i -> low byte of BE halfword i = byte 2i+1), the per-lane VMLH multiplier
// (2^p, big-endian 16-bit), and the three VPERM gather controls.
func constants() (spread, mul, g0, g1, g2 []byte) {
	spread = make([]byte, 16)
	mul = make([]byte, 16)
	for x := range spread {
		spread[x] = 0x1f // index into all-zero second operand -> 0
	}
	for i := 0; i < 8; i++ {
		_, _, p := fieldInfo(i)
		spread[2*i+1] = byte(i)
		m := uint16(1) << uint(p)
		mul[2*i] = byte(m >> 8) // big-endian 16-bit multiplier
		mul[2*i+1] = byte(m)
	}
	src := make([][]int, 5)
	for i := 0; i < 8; i++ {
		hb, lb, _ := fieldInfo(i)
		src[hb] = append(src[hb], 2*i)      // high byte of window
		if lb != hb {
			src[lb] = append(src[lb], 2*i+1) // low byte of window
		}
	}
	mk := func() []byte {
		b := make([]byte, 16)
		for x := range b {
			b[x] = 0x1f
		}
		return b
	}
	g0, g1, g2 = mk(), mk(), mk()
	gs := [3][]byte{g0, g1, g2}
	for j := 0; j < 5; j++ {
		for k, s := range src[j] {
			gs[k][j] = byte(s)
		}
	}
	return
}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Int64)},
	)
}

func main() {
	f := emit.NewFile("s390x")

	spreadB, mulB, g0B, g1B, g2B := constants()
	spread := f.Data("spread", spreadB)
	mul := f.Data("mul", mulB)
	g0 := f.Data("g0", g0B)
	g1 := f.Data("g1", g1B)
	g2 := f.Data("g2", g2B)
	cA := f.Data("cA", repByte('A', 16))
	c25 := f.Data("c25", repByte(25, 16))
	c2 := f.Data("c2", repByte('2', 16))
	c5 := f.Data("c5", repByte(5, 16))
	c65 := f.Data("c65", repByte(65, 16))
	c24 := f.Data("c24", repByte(24, 16))

	b := s390x.NewFunc("decodeBlocksVX", sig(), 0)
	b.LoadArg("dst_base", "R1").LoadArg("src_base", "R2").LoadArg("n", "R3").
		Raw("MOVD $%s(SB), R4", spread).Raw("VL (R4), V8").
		Raw("MOVD $%s(SB), R4", mul).Raw("VL (R4), V9").
		Raw("MOVD $%s(SB), R4", g0).Raw("VL (R4), V10").
		Raw("MOVD $%s(SB), R4", g1).Raw("VL (R4), V11").
		Raw("MOVD $%s(SB), R4", g2).Raw("VL (R4), V12").
		Raw("MOVD $%s(SB), R4", cA).Raw("VL (R4), V13").
		Raw("MOVD $%s(SB), R4", c25).Raw("VL (R4), V14").
		Raw("MOVD $%s(SB), R4", c2).Raw("VL (R4), V15").
		Raw("MOVD $%s(SB), R4", c5).Raw("VL (R4), V16").
		Raw("MOVD $%s(SB), R4", c65).Raw("VL (R4), V17").
		Raw("MOVD $%s(SB), R4", c24).Raw("VL (R4), V18").
		Raw("VZERO V19").    // zero vector
		Raw("MOVD $0, R5").  // blocks decoded
		Raw("CMPBEQ R3, $0, done").
		Label("loop").
		Raw("VL (R2), V0"). // V0 = 8 chars in bytes 0..7 (element 0 = lowest addr)
		// validate: az = NOT(c-'A' >u 25); tw = NOT(c-'2' >u 5)
		Raw("VSB V13, V0, V1"). // V1 = c-'A'   (VSB VRT,VRA,VRB: VRT=VRA-VRB? pinned by test)
		Raw("VCHLB V1, V14, V1").Raw("VNO V1, V1, V1"). // az
		Raw("VSB V15, V0, V2"). // V2 = c-'2'
		Raw("VCHLB V2, V16, V2").Raw("VNO V2, V2, V2"). // tw
		Raw("VO V1, V2, V3"). // valid per lane
		// block-valid: extract high doubleword (lanes 0..7) to GPR, require -1
		Raw("VLGVG $0, V3, R6").
		Raw("CMPBNE R6, $-1, done").
		// map: value = (c-65)&az | (c-24)&tw
		Raw("VSB V17, V0, V4").Raw("VN V4, V1, V4"). // (c-65)&az
		Raw("VSB V18, V0, V0").Raw("VN V0, V2, V0"). // (c-24)&tw
		Raw("VO V4, V0, V0").                        // V0 = 8 values bytes 0..7
		// spread to low byte of each BE halfword, multiply-low by 2^p (= << p).
		// VMLHW is the Go mnemonic for VECTOR MULTIPLY LOW with halfword elements
		// (the bare VMLH is multiply-HIGH-logical — not what we want here).
		Raw("VPERM V0, V19, V8, V0").
		Raw("VMLHW V0, V9, V0").
		// gather to 5 output bytes
		Raw("VPERM V0, V19, V10, V5").
		Raw("VPERM V0, V19, V11, V6").Raw("VO V5, V6, V5").
		Raw("VPERM V0, V19, V12, V6").Raw("VO V5, V6, V5").
		Raw("VST V5, (R1)"). // store 16 bytes (caller keeps low 5)
		Raw("ADD $8, R2").Raw("ADD $5, R1").Raw("ADD $1, R5").
		Raw("ADD $-1, R3").Raw("CMPBNE R3, $0, loop").
		Label("done").Raw("MOVD R5, ret+56(FP)").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("decode_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_s390x.s")
}
