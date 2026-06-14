//go:build ignore

// Command gen produces encode_ppc64le.s with go-asmgen: a vectorised base32
// (RFC 4648 StdEncoding) encoder for POWER8+ VSX. VSX is baseline on ppc64le,
// so there is no runtime feature dispatch — the kernel always runs.
//
// Algorithm (per 5-byte group -> 8 chars), mirroring the amd64 SSE path but with
// VSX equivalents:
//
//  1. VPERM (with the `shuf` control vector) spreads the 16-byte input into eight
//     16-bit halfword lanes, each lane holding the big-endian 16-bit window that
//     contains one output char's 5-bit field.
//  2. VSRH performs a *per-lane variable* right shift by `shcnt[i] = p` so each
//     char's 5-bit field drops to bits [4:0] of its halfword. (amd64 has no
//     per-lane variable shift, so it fakes it with PMULHUW by 2^(16-p); ppc64le's
//     VSX exposes the variable shift directly — one of the two ops the Go arm64
//     assembler lacks that blocked the NEON port.)
//  3. VPERM (`pack`) gathers the low byte of each halfword (byte index 2i+1) into
//     the low 8 bytes; VAND 0x1f isolates the value 0..31.
//  4. Two-range ASCII map: v<26 -> 'A'+v, v>=26 -> '2'+(v-26), computed as
//     v + 65 - (VCMPGTUB(v,25) & 41).
//  5. STXVB16X stores the 16-byte vector; the caller consumes the low 8 chars.
//
// Endianness / VSX↔VMX aliasing: loads use LXVB16X, which gives big-endian
// *element* semantics (memory byte 0 == element 0's most-significant byte), so
// the VPERM control vectors and the multiplier/shift tables are laid out in
// big-endian lane order (lane 0 = lowest address). LXVB16X writes a VS register
// (VS32..), and the VMX arithmetic (VPERM/VSRH/...) names the SAME physical
// register as V0.. (Vn == VS(32+n)); we load into VS40.. and use V8.. etc.
// LXVB16X/STXVB16X are used (not LXVD2X) so no per-doubleword swap is needed.
// The exact lane mapping is pinned by a position-dependent qemu fuzz test.
//
// Run: go run encode_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

// fieldInfo returns, for output char i, the input byte holding the high part of
// its 5-bit field, the byte holding the low part, and the bit offset p of the
// field's LSB within the big-endian 16-bit window [hiByte:loByte].
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
// (`shuf`), the per-lane shift-count vector (`shcnt`), and the VPERM pack control
// (`pack`).
func constants() (shuf, shcnt, pack []byte) {
	shuf = make([]byte, 16)
	shcnt = make([]byte, 16)
	pack = make([]byte, 16)
	for i := 0; i < 8; i++ {
		hb, lb, p := fieldInfo(i)
		shuf[2*i] = byte(hb) // BE halfword i: high byte <- src[hiByte]
		shuf[2*i+1] = byte(lb)
		shcnt[2*i] = byte(p) // both bytes of the lane hold p (only low nibble used)
		shcnt[2*i+1] = byte(p)
		pack[i] = byte(2*i + 1) // value byte = low byte of halfword i
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
	f := emit.NewFile("ppc64le")

	shufB, shcntB, packB := constants()
	shuf := f.Data("shuf", shufB)
	shcnt := f.Data("shcnt", shcntB)
	pack := f.Data("pack", packB)
	mask1f := f.Data("mask1f", repByte(0x1f, 16))
	c25 := f.Data("c25", repByte(25, 16))
	c65 := f.Data("c65", repByte(65, 16))
	c41 := f.Data("c41", repByte(41, 16))

	b := ppc64.NewFunc("encodeBlocksVSX", sig(), 0)
	b.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").LoadArg("n", "R5").
		// Load constant vectors into VS40..VS46 (== V8..V14).
		Raw("MOVD $%s(SB), R6", shuf).Raw("LXVB16X (R6), VS40").
		Raw("MOVD $%s(SB), R6", shcnt).Raw("LXVB16X (R6), VS41").
		Raw("MOVD $%s(SB), R6", pack).Raw("LXVB16X (R6), VS42").
		Raw("MOVD $%s(SB), R6", c25).Raw("LXVB16X (R6), VS43").
		Raw("MOVD $%s(SB), R6", c65).Raw("LXVB16X (R6), VS44").
		Raw("MOVD $%s(SB), R6", c41).Raw("LXVB16X (R6), VS45").
		Raw("MOVD $%s(SB), R6", mask1f).Raw("LXVB16X (R6), VS46").
		Raw("CMP R5, $0").Raw("BEQ done").
		Label("loop").
		Raw("LXVB16X (R4), VS32").    // V0 = src[0:16] (BE element semantics)
		Raw("VPERM V0, V0, V8, V1").  // spread -> 8 BE halfword windows
		Raw("VSRH V1, V9, V1").       // per-lane >> p : field to bits[4:0]
		Raw("VPERM V1, V1, V10, V4"). // pack value bytes into low 8
		Raw("VAND V4, V14, V4").      // & 0x1f
		Raw("VCMPGTUB V4, V11, V5").  // v>25 ? 0xff : 0
		Raw("VAND V5, V13, V5").      // & 41
		Raw("VADDUBM V4, V12, V4").   // + 65
		Raw("VSUBUBM V4, V5, V4").    // V4 = V4 - V5   (vsububm VRT,VRA,VRB: VRT=VRA-VRB)
		Raw("STXVB16X VS36, (R3)").   // store 8 chars (16-byte store; caller uses 8)
		Raw("ADD $5, R4").Raw("ADD $8, R3").
		Raw("ADD $-1, R5").Raw("CMP R5, $0").Raw("BNE loop").
		Label("done").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("encode_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_ppc64le.s")
}
