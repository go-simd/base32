//go:build ppc64le || s390x

package base32

import (
	stdb32 "encoding/base32"
	"math/rand"
	"testing"
)

// TestEncodeVecKernel drives the VSX/vector kernel through encodeSIMD across many
// lengths and alignments, finishing the tail with encoding/base32 exactly as the
// public Encode does, and compares byte-for-byte against the stdlib. This runs
// the SIMD path even on the (qemu) build where it is the only kernel.
func TestEncodeVecKernel(t *testing.T) {
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

// TestEncodeSIMDDispatch exercises every branch of encodeSIMD: the n<16 early
// return, the normal multi-group path, and — by passing a deliberately tight dst
// — the group-capping loop that protects the final 16-byte vector store, down to
// the groups==0 fallback. The capped result, combined with the stdlib tail in
// Encode, must still be byte-identical to encoding/base32.
func TestEncodeSIMDDispatch(t *testing.T) {
	rng := rand.New(rand.NewSource(11))

	// n<16: SIMD does nothing.
	if sd, dd := encodeSIMD(make([]byte, 64), make([]byte, 10)); sd != 0 || dd != 0 {
		t.Fatalf("n<16: want (0,0), got (%d,%d)", sd, dd)
	}

	// Tight dst forces the capping loop. For each n>=16, shrink dst from its
	// natural EncodedLen down to (and below) the point where no full 16-byte
	// store fits, checking the consumed/written counts stay self-consistent and
	// that a full Encode into an exact buffer is still correct.
	for _, n := range []int{16, 17, 20, 21, 30, 64} {
		src := make([]byte, n)
		rng.Read(src)
		full := EncodedLen(n)
		for cap := 0; cap <= full; cap++ {
			dst := make([]byte, cap)
			sd, dd := encodeSIMD(dst, src)
			if dd != sd/5*8 || sd%5 != 0 {
				t.Fatalf("n=%d cap=%d: inconsistent (sd=%d,dd=%d)", n, cap, sd, dd)
			}
			if (sd > 0) && (dd-8+16 > cap) {
				t.Fatalf("n=%d cap=%d: last store overruns dst (dd=%d)", n, cap, dd)
			}
		}
		// Exact-size buffer through the public API must match the stdlib.
		dst := make([]byte, full)
		Encode(dst, src)
		if want := stdb32.StdEncoding.EncodeToString(src); string(dst) != want {
			t.Fatalf("n=%d exact:\n got=%q\nwant=%q", n, string(dst), want)
		}
	}
}
