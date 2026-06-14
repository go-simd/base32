//go:build ignore

// Command gen produces decode_ppc64le.s with go-asmgen: a vectorised base32
// (RFC 4648 StdEncoding) decoder for POWER8+ VSX. VSX is baseline on ppc64le, so
// there is no runtime feature dispatch — the kernel always runs.
//
// The decoder is the inverse of the encoder. Per 8-char block -> 5 bytes,
// mirroring the amd64 SSE decode path with VSX equivalents:
//
//  1. Validate + map ASCII -> 5-bit value, per lane, with two unsigned range
//     checks. az = c in 'A'..'Z' = NOT(c-'A' >u 25); tw = c in '2'..'7' =
//     NOT(c-'2' >u 5), using VCMPGTUB + VNOR. value = VSEL between (c-65) and
//     (c-24) by the az mask; lanes matching neither range are invalid. The block
//     is decoded only if all 8 active lanes are valid (checked by AND-reducing
//     the per-lane valid mask to a GPR via MFVSRD). The kernel counts consumed
//     blocks and stops before the first invalid block.
//  2. VPERM (`spread`) scatters the eight value bytes into the low byte of each
//     big-endian 16-bit lane.
//  3. VSLH performs a *per-lane variable* left shift by shcnt[i]=p, placing each
//     value into its big-endian 16-bit output window — the exact inverse of the
//     encoder's VSRH right shift (POWER exposes the variable shift natively).
//  4. Three VPERM gathers + VOR scatter each window's high/low byte into the five
//     output bytes (each takes up to three overlapping contributions).
//  5. STXVB16X stores the 16-byte vector; the caller keeps the low 5 bytes.
//
// Endianness / VSX↔VMX aliasing follows the encoder: LXVB16X (big-endian element
// semantics, byte 0 = element 0's MSB) loads into VS(32+k), arithmetic names Vk.
// The lane mapping is pinned by a position-dependent qemu fuzz test.
//
// Run: go run decode_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
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
// byte i -> low byte of BE halfword i, i.e. byte 2i+1), the per-lane shift-count
// vector (shcnt[i]=p in both bytes of the halfword), and the three VPERM gather
// controls (output byte <- one source byte of the shifted vector; 0x10.. selects
// a zero byte from the second VPERM operand).
func constants() (spread, shcnt, g0, g1, g2 []byte) {
	spread = make([]byte, 16)
	shcnt = make([]byte, 16)
	for x := range spread {
		spread[x] = 0x1f // index into all-zero second operand -> 0
	}
	for i := 0; i < 8; i++ {
		_, _, p := fieldInfo(i)
		// value byte i is at memory byte i (BE element i) -> place into low byte of
		// halfword i = byte 2i+1; high byte 2i stays zero.
		spread[2*i+1] = byte(i)
		shcnt[2*i] = byte(p)
		shcnt[2*i+1] = byte(p)
	}
	// gather: after VSLH, BE halfword i occupies bytes [2i (hi), 2i+1 (lo)].
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
			b[x] = 0x1f // -> zero
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
	f := emit.NewFile("ppc64le")

	spreadB, shcntB, g0B, g1B, g2B := constants()
	spread := f.Data("spread", spreadB)
	shcnt := f.Data("shcnt", shcntB)
	g0 := f.Data("g0", g0B)
	g1 := f.Data("g1", g1B)
	g2 := f.Data("g2", g2B)
	cA := f.Data("cA", repByte('A', 16))
	c25 := f.Data("c25", repByte(25, 16))
	c2 := f.Data("c2", repByte('2', 16))
	c5 := f.Data("c5", repByte(5, 16))
	c65 := f.Data("c65", repByte(65, 16))
	c24 := f.Data("c24", repByte(24, 16))

	b := ppc64.NewFunc("decodeBlocksVSX", sig(), 0)
	b.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").LoadArg("n", "R5").
		// Constant vectors into VS40..VS50 (== V8..V18). Zero vector in V19.
		Raw("MOVD $%s(SB), R6", spread).Raw("LXVB16X (R6), VS40"). // V8
		Raw("MOVD $%s(SB), R6", shcnt).Raw("LXVB16X (R6), VS41").  // V9
		Raw("MOVD $%s(SB), R6", g0).Raw("LXVB16X (R6), VS42").     // V10
		Raw("MOVD $%s(SB), R6", g1).Raw("LXVB16X (R6), VS43").     // V11
		Raw("MOVD $%s(SB), R6", g2).Raw("LXVB16X (R6), VS44").     // V12
		Raw("MOVD $%s(SB), R6", cA).Raw("LXVB16X (R6), VS45").     // V13
		Raw("MOVD $%s(SB), R6", c25).Raw("LXVB16X (R6), VS46").    // V14
		Raw("MOVD $%s(SB), R6", c2).Raw("LXVB16X (R6), VS47").     // V15
		Raw("MOVD $%s(SB), R6", c5).Raw("LXVB16X (R6), VS48").     // V16
		Raw("MOVD $%s(SB), R6", c65).Raw("LXVB16X (R6), VS49").    // V17
		Raw("MOVD $%s(SB), R6", c24).Raw("LXVB16X (R6), VS50").    // V18
		Raw("VSPLTISB $0, V19").                                   // V19 = zero
		Raw("MOVD $0, R7").                                        // R7 = blocks decoded
		Raw("CMP R5, $0").Raw("BEQ done").
		Label("loop").
		Raw("LXVB16X (R4), VS32"). // V0 = 8 chars in bytes 0..7 (BE elements)
		// validate: az = NOT(c-'A' >u 25); tw = NOT(c-'2' >u 5)
		Raw("VSUBUBM V0, V13, V1"). // V1 = c-'A'
		Raw("VCMPGTUB V1, V14, V1").Raw("VNOR V1, V1, V1"). // V1 = az (0xff if A..Z)
		Raw("VSUBUBM V0, V15, V2"). // V2 = c-'2'
		Raw("VCMPGTUB V2, V16, V2").Raw("VNOR V2, V2, V2"). // V2 = tw
		Raw("VOR V1, V2, V3"). // V3 = valid per lane
		// block-valid check: AND-reduce the 8 active lanes (bytes 0..7 = high
		// doubleword) to a GPR and require all ones.
		Raw("MFVSRD VS35, R8"). // R8 = high doubleword of V3 (bytes 0..7)
		Raw("CMP R8, $-1").     // all 8 lanes 0xff -> 0xffffffffffffffff (== -1 signed) ?
		Raw("BNE done").
		// map: value = (c-65)&az | (c-24)&tw
		Raw("VSUBUBM V0, V17, V4").Raw("VAND V4, V1, V4"). // (c-65)&az
		Raw("VSUBUBM V0, V18, V0").Raw("VAND V0, V2, V0"). // (c-24)&tw
		Raw("VOR V4, V0, V0").                             // V0 = 8 values, bytes 0..7
		// spread to low byte of each BE halfword, then variable left shift by p
		Raw("VPERM V0, V19, V8, V0"). // V0 = spread (second operand zero)
		Raw("VSLH V0, V9, V0").       // per-lane << p
		// gather to 5 output bytes (3 VPERM + VOR; second operand zero -> 0x10.. = 0)
		Raw("VPERM V0, V19, V10, V5").
		Raw("VPERM V0, V19, V11, V6").Raw("VOR V5, V6, V5").
		Raw("VPERM V0, V19, V12, V6").Raw("VOR V5, V6, V5").
		Raw("STXVB16X VS37, (R3)"). // store 16 bytes (caller keeps low 5)
		Raw("ADD $8, R4").Raw("ADD $5, R3").Raw("ADD $1, R7").
		Raw("ADD $-1, R5").Raw("CMP R5, $0").Raw("BNE loop").
		Label("done").Raw("MOVD R7, ret+56(FP)").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("decode_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_ppc64le.s")
}
