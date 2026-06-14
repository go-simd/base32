//go:build arm64 && go1.27

package base32

import (
	stdb32 "encoding/base32"
	"math/rand"
	"testing"
)

// TestEncodeNEONKernel drives the NEON kernel through encodeSIMD across many
// lengths and alignments, finishing the tail with encoding/base32 exactly as the
// public Encode does, and compares byte-for-byte against the stdlib. This runs
// the SIMD path on every native arm64 / Go 1.27 host.
func TestEncodeNEONKernel(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{0, 1, 4, 5, 9, 10, 15, 16, 17, 20, 21, 25, 30, 31, 32, 33, 100, 1000, 4096} {
		src := make([]byte, n)
		rng.Read(src)
		dst := make([]byte, EncodedLen(n))
		Encode(dst, src)
		want := stdb32.StdEncoding.EncodeToString(src)
		if string(dst) != want {
			t.Fatalf("n=%d:\n got=%q\nwant=%q", n, string(dst), want)
		}
	}
}

// TestEncodeSIMDDispatchNEON exercises every branch of encodeSIMD: the n<16
// early return, the normal multi-group path, and — by passing a deliberately
// tight dst — the group-capping loop that protects the final 16-byte vector
// store, down to the groups==0 fallback. The capped result, combined with the
// stdlib tail in Encode, must still be byte-identical to encoding/base32.
func TestEncodeSIMDDispatchNEON(t *testing.T) {
	rng := rand.New(rand.NewSource(11))

	// n<16: SIMD does nothing.
	if sd, dd := encodeSIMD(make([]byte, 64), make([]byte, 10)); sd != 0 || dd != 0 {
		t.Fatalf("n<16: want (0,0), got (%d,%d)", sd, dd)
	}

	for _, n := range []int{16, 17, 20, 21, 30, 64} {
		src := make([]byte, n)
		rng.Read(src)
		full := EncodedLen(n)
		for capLen := 0; capLen <= full; capLen++ {
			dst := make([]byte, capLen)
			sd, dd := encodeSIMD(dst, src)
			if dd != sd/5*8 || sd%5 != 0 {
				t.Fatalf("n=%d cap=%d: inconsistent (sd=%d,dd=%d)", n, capLen, sd, dd)
			}
			if (sd > 0) && (dd-8+16 > capLen) {
				t.Fatalf("n=%d cap=%d: last store overruns dst (dd=%d)", n, capLen, dd)
			}
		}
		dst := make([]byte, full)
		Encode(dst, src)
		if want := stdb32.StdEncoding.EncodeToString(src); string(dst) != want {
			t.Fatalf("n=%d exact:\n got=%q\nwant=%q", n, string(dst), want)
		}
	}
}

// TestDecodeSIMDNoopNEON pins arm64's no-op decodeSIMD: there is no NEON decode
// kernel, so it must always report (0,0) and leave the whole input to the stdlib
// (covered for the coverage gate).
func TestDecodeSIMDNoopNEON(t *testing.T) {
	if sd, dd := decodeSIMD(make([]byte, 64), make([]byte, 64)); sd != 0 || dd != 0 {
		t.Fatalf("decodeSIMD: want (0,0), got (%d,%d)", sd, dd)
	}
}

func benchEncodeNEON(b *testing.B) {
	src := make([]byte, 1<<20)
	rand.New(rand.NewSource(2)).Read(src)
	dst := make([]byte, EncodedLen(len(src)))
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(dst, src)
	}
}

func BenchmarkEncodeNEON(b *testing.B) { benchEncodeNEON(b) }
