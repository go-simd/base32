//go:build ignore

// Command gen produces encode_s390x.s with go-asmgen: a vectorised base32
// (RFC 4648 StdEncoding) encoder for z13+ (the vector facility is baseline
// there, so there is no runtime feature dispatch — the kernel always runs).
//
// Algorithm (per 5-byte group -> 8 chars), a faithful port of the amd64 SSE
// path, including its multiply-high trick — s390x is the one shipped non-amd64
// arch with a genuine vector integer multiply-high (VMLHH), so it reproduces the
// amd64 kernel almost instruction-for-instruction (this is exactly the op the Go
// arm64 assembler lacks, which blocked the NEON port):
//
//  1. VPERM (`shuf`) spreads the 16-byte input into eight 16-bit halfword lanes,
//     each lane holding the big-endian 16-bit window containing one output char's
//     5-bit field.
//  2. VMLHH (Vector Multiply Logical High, halfword = unsigned high 16 bits of
//     the 16x16 product) by the per-lane multiplier 2^(16-p) shifts each char's
//     5-bit field down to bits [4:0] — exactly amd64's PMULHUW.
//  3. VPERM (`pack`) gathers the low byte of each halfword into the low 8 bytes;
//     VN 0x1f isolates the value 0..31.
//  4. Two-range ASCII map: v<26 -> 'A'+v, v>=26 -> '2'+(v-26), computed as
//     v + 65 - (VCHLB(v,25) & 41), where VCHLB is unsigned compare-high byte.
//  5. VST stores the 16-byte vector; the caller consumes the low 8 chars.
//
// BIG-ENDIAN: s390x is the only big-endian target. VL puts the LOWEST memory
// address into element 0 (the high-order lane), and VST writes element 0 to the
// lowest address. So the VPERM control vectors and the multiplier table are laid
// out in big-endian lane order (lane 0 = lowest address) — the same byte/lane
// layout the (little-endian) amd64 path uses, because amd64's PSHUFB indexes by
// byte position and its 16-bit windows are themselves big-endian. No scan-order
// reversal is needed; the cross-lane shuffle is pinned by a position-dependent
// qemu fuzz test.
//
// Run: go run encode_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

// fieldInfo returns, for output char i, the input byte holding the high part of
// its 5-bit field, the byte holding the low part, and the bit offset p of the
// field's LSB within the big-endian 16-bit window [hiByte:loByte]. The VMLHH
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

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

// constants builds, in big-endian halfword-lane order, the VPERM spread control
// (`shuf`), the per-lane VMLHH multiplier table (`mul`, eight big-endian 16-bit
// values), and the VPERM pack control (`pack`).
func constants() (shuf, mul, pack []byte) {
	shuf = make([]byte, 16)
	mul = make([]byte, 16)
	pack = make([]byte, 16)
	for i := 0; i < 8; i++ {
		hb, lb, p := fieldInfo(i)
		shuf[2*i] = byte(hb) // BE halfword i: high byte <- src[hiByte]
		shuf[2*i+1] = byte(lb)
		m := uint16(1) << uint(16-p)
		mul[2*i] = byte(m >> 8) // big-endian 16-bit multiplier
		mul[2*i+1] = byte(m)
		pack[i] = byte(2*i + 1) // value byte = low byte of halfword i's high product
	}
	for i := 8; i < 16; i++ {
		pack[i] = 0 // upper 8 result bytes: harmless, masked away later
	}
	return
}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		nil,
	)
}

func main() {
	f := emit.NewFile("s390x")

	shufB, mulB, packB := constants()
	shuf := f.Data("shuf", shufB)
	mul := f.Data("mul", mulB)
	pack := f.Data("pack", packB)
	mask1f := f.Data("mask1f", repByte(0x1f, 16))
	c25 := f.Data("c25", repByte(25, 16))
	c65 := f.Data("c65", repByte(65, 16))
	c41 := f.Data("c41", repByte(41, 16))

	b := s390x.NewFunc("encodeBlocksVX", sig(), 0)
	b.LoadArg("dst_base", "R1").LoadArg("src_base", "R2").LoadArg("n", "R3").
		Raw("MOVD $%s(SB), R4", shuf).Raw("VL (R4), V8").
		Raw("MOVD $%s(SB), R4", mul).Raw("VL (R4), V9").
		Raw("MOVD $%s(SB), R4", pack).Raw("VL (R4), V10").
		Raw("MOVD $%s(SB), R4", c25).Raw("VL (R4), V11").
		Raw("MOVD $%s(SB), R4", c65).Raw("VL (R4), V12").
		Raw("MOVD $%s(SB), R4", c41).Raw("VL (R4), V13").
		Raw("MOVD $%s(SB), R4", mask1f).Raw("VL (R4), V14").
		Raw("CMPBEQ R3, $0, done").
		Label("loop").
		Raw("VL (R2), V0").           // V0 = src[0:16] (element 0 = lowest addr)
		Raw("VPERM V0, V0, V8, V1").  // spread -> 8 BE halfword windows
		Raw("VMLHH V1, V9, V1").      // unsigned high product = field >> p
		Raw("VPERM V1, V1, V10, V4"). // pack value bytes into low 8
		Raw("VN V4, V14, V4").        // & 0x1f
		Raw("VCHLB V4, V11, V5").     // v>25 ? 0xff : 0 (unsigned compare high)
		Raw("VN V5, V13, V5").        // & 41
		Raw("VAB V4, V12, V4").       // + 65
		Raw("VSB V5, V4, V4").        // V4 = V4 - V5
		Raw("VST V4, (R1)").          // store 8 chars (16-byte store; caller uses 8)
		Raw("ADD $5, R2").Raw("ADD $8, R1").
		Raw("ADD $-1, R3").Raw("CMPBNE R3, $0, loop").
		Label("done").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("encode_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_s390x.s")
}
