//go:build ignore

// Command gen produces encode_arm64.s with go-asmgen: a vectorised base32
// (RFC 4648 StdEncoding) encoder for arm64 NEON. It is a faithful port of the
// amd64 SSE path (spread / multiply-high / mask / alphabet-map), and is the
// concrete demonstration of the three NEON ops the *released* Go arm64
// assembler does not expose but Go master / Go 1.27 does — the same gap the
// ppc64le (VSRH) and s390x (VMLHH) ports already worked around:
//
//   - VUMULL / VUMULL2 : the integer *widening* vector multiply (16x16 -> 32).
//     Released Go only assembled the polynomial VPMULL; the integer multiply
//     mnemonics landed upstream in Go 1.27.
//   - USHL : the per-lane *register-variable* shift (the shift count comes from
//     a vector register, one count per lane), used here to take the high half
//     of each 32-bit product (multiply-high == amd64 PMULHUW).
//   - VTBL : the table-lookup permute, used twice — once to spread the 5-byte
//     group into eight big-endian 16-bit windows, once as the 32-entry base32
//     alphabet LUT (a two-register table, which covers indices 0..31).
//
// Because these are //go:build go1.27-only, the generated kernel is guarded
// //go:build arm64 && go1.27 and the stable build falls back to the standard
// library via encode_generic.go.
//
// Algorithm (per 5-byte group -> 8 chars):
//
//  1. VTBL (`spread`) gathers the 16-byte input into eight 16-bit halfword
//     lanes, each lane holding the big-endian 16-bit window containing one
//     output char's 5-bit field.
//  2. VUMULL/VUMULL2 multiply each 16-bit window by the per-lane constant
//     2^(16-p) into 32-bit products; USHL by -16 (a register count vector)
//     keeps the high 16 bits, i.e. the field shifted down to bits [4:0]
//     (multiply-high). VAND 0x1f isolates the value 0..31.
//  3. VTBL (`pack`) gathers the low byte of each halfword (byte index 2i) into
//     the low 8 bytes — note big-endian windows put the field's value byte in
//     the high (even) byte of the 16-bit high product after the >>16.
//  4. VTBL with the two-register base32 alphabet table maps each value 0..31 to
//     its ASCII char in one lookup (no two-range arithmetic needed).
//  5. VST1 stores the 16-byte vector; the caller consumes the low 8 chars.
//
// arm64 is little-endian and NEON loads/stores are in element order with the
// lowest address in lane 0, exactly like the amd64 SSE path; the 16-bit windows
// are themselves big-endian, so the VTBL control vectors and the multiplier
// table use the same byte/lane layout the amd64 path uses.
//
// Run: go run encode_arm64_gen.go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

// fieldInfo returns, for output char i, the input byte holding the high part of
// its 5-bit field, the byte holding the low part, and the bit offset p of the
// field's LSB within the big-endian 16-bit window [hiByte:loByte]. The
// multiply-high multiplier for that lane is 2^(16-p).
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

// constants builds, in big-endian halfword-lane order, the VTBL spread control
// (`spread`), the per-lane multiply-high multiplier table (`mul`, eight 16-bit
// values), the VTBL pack control (`pack`) and the per-lane -16 shift counts.
func constants() (spread, mul, pack, sh []byte) {
	spread = make([]byte, 16)
	mul = make([]byte, 16)
	pack = make([]byte, 16)
	sh = make([]byte, 16)
	for i := 0; i < 8; i++ {
		hb, lb, p := fieldInfo(i)
		// Big-endian 16-bit window of halfword lane i: low address (even byte
		// of the lane) holds the high source byte. NEON lane i occupies bytes
		// 2i (low) and 2i+1 (high) of the vector; a little-endian H lane reads
		// byte 2i as bits[7:0] and 2i+1 as bits[15:8], so to build a value whose
		// numeric high byte is src[hb] we put src[hb] at byte 2i+1 and src[lb]
		// at byte 2i.
		spread[2*i] = byte(lb)
		spread[2*i+1] = byte(hb)
		m := uint16(1) << uint(16-p)
		mul[2*i] = byte(m)      // little-endian 16-bit multiplier
		mul[2*i+1] = byte(m >> 8)
		// After USHL -16 the field value sits in bits[4:0] of halfword lane i,
		// i.e. the low (even) byte 2i of the vector.
		pack[i] = byte(2 * i)
		sh[2*i] = 0xf0 // -16 as signed byte (USHL count, replicated per byte)
		sh[2*i+1] = 0xf0
	}
	for i := 8; i < 16; i++ {
		pack[i] = 0x1f // upper 8 result bytes -> spare value 0 lane; masked away
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
	f := emit.NewFile("arm64")

	spreadB, mulB, packB, shB := constants()
	spread := f.Data("spread", spreadB)
	mul := f.Data("mul", mulB)
	pack := f.Data("pack", packB)
	sh := f.Data("sh", shB)
	mask1f := f.Data("mask1f", repByte(0x1f, 16))
	// 32-entry base32 alphabet as a two-register VTBL table.
	alpha := []byte("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
	alphaLo := f.Data("alphalo", alpha[:16])
	alphaHi := f.Data("alphahi", alpha[16:])

	b := arm64.NewFunc("encodeBlocksNEON", sig(), 0)
	b.LoadArg("dst_base", "R0").LoadArg("src_base", "R1").LoadArg("n", "R2").
		// Load constant vectors.
		Raw("MOVD $%s(SB), R3", spread).Raw("VLD1 (R3), [V8.B16]").
		Raw("MOVD $%s(SB), R3", mul).Raw("VLD1 (R3), [V9.B16]").
		Raw("MOVD $%s(SB), R3", pack).Raw("VLD1 (R3), [V10.B16]").
		Raw("MOVD $%s(SB), R3", sh).Raw("VLD1 (R3), [V11.B16]").
		Raw("MOVD $%s(SB), R3", mask1f).Raw("VLD1 (R3), [V12.B16]").
		// Two-register alphabet table in consecutive vector regs V13,V14.
		Raw("MOVD $%s(SB), R3", alphaLo).Raw("VLD1 (R3), [V13.B16]").
		Raw("MOVD $%s(SB), R3", alphaHi).Raw("VLD1 (R3), [V14.B16]").
		Raw("CBZ R2, done").
		Label("loop").
		Raw("VLD1 (R1), [V0.B16]"). // V0 = src[0:16]
		// Spread into eight big-endian 16-bit windows (VTBL: V8 indexes V0).
		Raw("VTBL V8.B16, [V0.B16], V1.B16").
		// Multiply-high: widen-multiply each 16-bit window by 2^(16-p), then >>16.
		Raw("VUMULL V9.H4, V1.H4, V20.S4").
		Raw("VUMULL2 V9.H8, V1.H8, V21.S4").
		// VUSHL by the register count vector (-16) -> high 16 bits into low half.
		Raw("VUSHL V11.S4, V20.S4, V20.S4").
		Raw("VUSHL V11.S4, V21.S4, V21.S4").
		// Narrow the two S4 halves back to one H8 (low 16 bits of each product).
		Raw("VXTN V20.S4, V2.H4").
		Raw("VXTN2 V21.S4, V2.H8").
		// & 0x1f -> value 0..31 in bits[4:0] of each halfword.
		Raw("VAND V12.B16, V2.B16, V2.B16").
		// Pack the value byte (even byte 2i) of each lane into the low 8 bytes.
		Raw("VTBL V10.B16, [V2.B16], V3.B16").
		// Alphabet LUT: map each value 0..31 to ASCII (two-register table).
		Raw("VTBL V3.B16, [V13.B16, V14.B16], V4.B16").
		Raw("VST1 [V4.B16], (R0)"). // store 8 chars (16-byte store; caller uses 8)
		Raw("ADD $5, R1").Raw("ADD $8, R0").
		Raw("SUB $1, R2").Raw("CBNZ R2, loop").
		Label("done").Ret()
	f.Add(b.Func())

	// VUMULL / USHL (integer) are only assemblable on Go 1.27+, matching the
	// //go:build go1.27 Go file; emit writes a bare "arm64", narrow it here.
	out := strings.Replace(f.String(), "//go:build arm64\n", "//go:build arm64 && go1.27\n", 1)
	if err := os.WriteFile("encode_arm64.s", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote encode_arm64.s")
}
