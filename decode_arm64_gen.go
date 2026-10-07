//go:build ignore

// Command gen produces decode_arm64.s with go-asmgen: a vectorised base32
// (RFC 4648 StdEncoding) decoder for arm64 NEON. NEON is baseline on arm64, so
// there is no runtime feature dispatch — the kernel always runs.
//
// The decoder is the inverse of the encoder (encode_arm64_gen.go). Per 8-char
// block -> 5 bytes, a port of the amd64 / s390x decode path:
//
//  1. Validate + map ASCII -> 5-bit value with ONE table lookup. d = c - '2'
//     (wrapping) puts every alphabet char in 0..40 ('2'..'7' -> 0..5, 'A'..'Z'
//     -> 15..40). A three-register VTBL (48 entries) maps d to 0x40|value for
//     an alphabet char and to 0 for anything else; VTBL itself yields 0 for an
//     index >= 48, so every non-alphabet byte (including '=') lands on 0.
//     VUMINV (unsigned minimum across lanes) then answers "are all chars of the
//     block(s) valid?" in one instruction: the minimum is >= 0x40 iff they are.
//     VAND 0x1f strips the flag, leaving the value 0..31.
//  2. VTBL (`spread`) places value i in the low byte of little-endian halfword
//     lane i (the high byte is 0: index 0xff is out of range).
//  3. VUSHL by the per-lane count p left-shifts each value into its 16-bit
//     output window — the inverse of the encoder's multiply-high right shift
//     (amd64 uses PMULLW by 2^p, s390x VMLHW, ppc64le VSLH).
//  4. Three VTBL gathers + VORR scatter each window's high/low byte into the
//     five output bytes (each output byte takes up to three overlapping
//     contributions).
//  5. VST1 stores a 16-byte vector; the caller keeps the decoded bytes and the
//     next store (or the stdlib tail) overwrites the rest.
//
// The main loop decodes TWO blocks per iteration (one 16-byte load, two-register
// VTBL gathers into 10 output bytes). When fewer than two blocks remain, or
// when the pair holds a non-alphabet char, a single-block step decodes the first
// block if it alone is valid, and the kernel stops: it returns the number of
// blocks decoded, stopping before the first block that contains a non-alphabet
// char, so the caller hands the remainder (errors, padding) to encoding/base32.
//
// arm64 is little-endian with lane 0 at the lowest address. A 16-bit window is
// big-endian in the source (the high byte comes first), so in halfword lane i
// the numeric high byte (vector byte 2i+1) belongs to output byte hiByte and the
// low byte (vector byte 2i) to output byte loByte — the same layout the encoder
// builds with its spread table.
//
// Run: go run decode_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
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

// lut is the 48-entry validate+map table indexed by c-'2': 0x40|value for an
// alphabet char, 0 otherwise.
func lut() []byte {
	t := make([]byte, 48)
	alpha := "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
	for v := 0; v < 32; v++ {
		t[alpha[v]-'2'] = 0x40 | byte(v)
	}
	return t
}

// constants builds the two spread controls (block A = value bytes 0..7, block B
// = value bytes 8..15), the per-halfword left-shift counts, and the three gather
// controls over the two-register table [windows A, windows B] (indices 16..31
// address block B, whose five bytes land at output 5..9).
func constants() (sprA, sprB, shl, g0, g1, g2 []byte) {
	sprA = repByte(0xff, 16)
	sprB = repByte(0xff, 16)
	shl = make([]byte, 16)
	for i := 0; i < 8; i++ {
		_, _, p := fieldInfo(i)
		sprA[2*i] = byte(i)
		sprB[2*i] = byte(8 + i)
		shl[2*i] = byte(p) // little-endian int16 shift count
	}
	src := make([][]int, 5)
	for i := 0; i < 8; i++ {
		hb, lb, _ := fieldInfo(i)
		src[hb] = append(src[hb], 2*i+1) // numeric high byte of window i
		if lb != hb {
			src[lb] = append(src[lb], 2*i) // numeric low byte of window i
		}
	}
	g0, g1, g2 = repByte(0xff, 16), repByte(0xff, 16), repByte(0xff, 16)
	gs := [3][]byte{g0, g1, g2}
	for j := 0; j < 5; j++ {
		for k, s := range src[j] {
			gs[k][j] = byte(s)
			gs[k][5+j] = byte(16 + s)
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
	f := emit.NewFile("arm64")

	sprAB, sprBB, shlB, g0B, g1B, g2B := constants()
	l := lut()
	c2 := f.Data("c2", repByte('2', 16))
	lut0 := f.Data("lut0", l[0:16])
	lut1 := f.Data("lut1", l[16:32])
	lut2 := f.Data("lut2", l[32:48])
	mask1f := f.Data("mask1f", repByte(0x1f, 16))
	sprA := f.Data("spra", sprAB)
	sprB := f.Data("sprb", sprBB)
	shl := f.Data("shl", shlB)
	g0 := f.Data("g0", g0B)
	g1 := f.Data("g1", g1B)
	g2 := f.Data("g2", g2B)

	b := arm64.NewFunc("decodeBlocksNEON", sig(), 0)
	b.LoadArg("dst_base", "R0").LoadArg("src_base", "R1").LoadArg("n", "R2").
		Raw("MOVD $%s(SB), R3", c2).Raw("VLD1 (R3), [V8.B16]").
		Raw("MOVD $%s(SB), R3", mask1f).Raw("VLD1 (R3), [V9.B16]").
		Raw("MOVD $%s(SB), R3", sprA).Raw("VLD1 (R3), [V10.B16]").
		Raw("MOVD $%s(SB), R3", sprB).Raw("VLD1 (R3), [V11.B16]").
		Raw("MOVD $%s(SB), R3", shl).Raw("VLD1 (R3), [V12.B16]").
		Raw("MOVD $%s(SB), R3", g0).Raw("VLD1 (R3), [V13.B16]").
		Raw("MOVD $%s(SB), R3", g1).Raw("VLD1 (R3), [V14.B16]").
		Raw("MOVD $%s(SB), R3", g2).Raw("VLD1 (R3), [V15.B16]").
		// Three-register validate+map table in consecutive regs V24..V26.
		Raw("MOVD $%s(SB), R3", lut0).Raw("VLD1 (R3), [V24.B16]").
		Raw("MOVD $%s(SB), R3", lut1).Raw("VLD1 (R3), [V25.B16]").
		Raw("MOVD $%s(SB), R3", lut2).Raw("VLD1 (R3), [V26.B16]").
		Raw("MOVD $0, R5"). // blocks decoded
		Label("pair").
		Raw("CMP $2, R2").Raw("BLT single").
		Raw("VLD1 (R1), [V0.B16]"). // two blocks: chars 0..15
		Raw("VSUB V8.B16, V0.B16, V0.B16").
		Raw("VTBL V0.B16, [V24.B16, V25.B16, V26.B16], V0.B16").
		Raw("VUMINV V0.B16, V1").Raw("VMOV V1.B[0], R6").
		Raw("CMP $0x40, R6").Raw("BLT single"). // a char of the pair is invalid
		Raw("VAND V9.B16, V0.B16, V0.B16").     // 16 values 0..31
		Raw("VTBL V10.B16, [V0.B16], V20.B16"). // block A values -> halfword lanes
		Raw("VTBL V11.B16, [V0.B16], V21.B16"). // block B values -> halfword lanes
		Raw("VUSHL V12.H8, V20.H8, V20.H8").    // value << p: output windows
		Raw("VUSHL V12.H8, V21.H8, V21.H8").
		Raw("VTBL V13.B16, [V20.B16, V21.B16], V2.B16").
		Raw("VTBL V14.B16, [V20.B16, V21.B16], V3.B16").
		Raw("VTBL V15.B16, [V20.B16, V21.B16], V4.B16").
		Raw("VORR V3.B16, V2.B16, V2.B16").
		Raw("VORR V4.B16, V2.B16, V2.B16").
		Raw("VST1 [V2.B16], (R0)"). // 16-byte store; 10 bytes decoded
		Raw("ADD $16, R1").Raw("ADD $10, R0").Raw("ADD $2, R5").
		Raw("SUB $2, R2").Raw("B pair").
		// One block: the last one, or the first of an invalid pair. Either way
		// the kernel stops after it.
		Label("single").
		Raw("CBZ R2, done").
		Raw("VLD1 (R1), [V0.B8]"). // chars 0..7
		Raw("VSUB V8.B8, V0.B8, V0.B8").
		Raw("VTBL V0.B8, [V24.B16, V25.B16, V26.B16], V0.B8").
		Raw("VUMINV V0.B8, V1").Raw("VMOV V1.B[0], R6").
		Raw("CMP $0x40, R6").Raw("BLT done").
		Raw("VAND V9.B16, V0.B16, V0.B16").
		Raw("VTBL V10.B16, [V0.B16], V20.B16").
		Raw("VUSHL V12.H8, V20.H8, V20.H8").
		// One-register gathers: indices 16..31 (block B) are out of range -> 0.
		Raw("VTBL V13.B16, [V20.B16], V2.B16").
		Raw("VTBL V14.B16, [V20.B16], V3.B16").
		Raw("VTBL V15.B16, [V20.B16], V4.B16").
		Raw("VORR V3.B16, V2.B16, V2.B16").
		Raw("VORR V4.B16, V2.B16, V2.B16").
		Raw("VST1 [V2.B16], (R0)"). // 16-byte store; 5 bytes decoded
		Raw("ADD $1, R5").
		Label("done").Raw("MOVD R5, ret+56(FP)").Ret()
	f.Add(b.Func())

	if err := os.WriteFile("decode_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote decode_arm64.s")
}
