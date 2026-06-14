//go:build ppc64le || s390x

package base32

import (
	stdb32 "encoding/base32"
	"math/rand"
	"testing"
)

// TestDecodeVecKernel drives the VSX/vector decode kernel through decodeSIMD
// across many lengths and alignments, finishing the remainder with
// encoding/base32 exactly as the public Decode does, and compares byte-for-byte
// against the stdlib. This runs the SIMD path even on the (qemu) build where it
// is the only kernel.
func TestDecodeVecKernel(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	for _, n := range []int{0, 1, 4, 5, 10, 15, 16, 17, 20, 21, 25, 30, 31, 32, 33, 100, 1000, 4096} {
		src := make([]byte, n)
		rng.Read(src)
		enc := []byte(stdb32.StdEncoding.EncodeToString(src))
		dst := make([]byte, DecodedLen(len(enc)))
		nn, err := Decode(dst, enc)
		if err != nil {
			t.Fatalf("n=%d: decode err: %v", n, err)
		}
		if string(dst[:nn]) != string(src) {
			t.Fatalf("n=%d: round-trip mismatch\n got=%x\nwant=%x", n, dst[:nn], src)
		}
	}
}

// TestDecodeVecInvalid drives the kernel's per-block validity bail and the
// error-offset shift, requiring byte-and-offset identical results to the stdlib.
func TestDecodeVecInvalid(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	for trial := 0; trial < 4000; trial++ {
		n := rng.Intn(60)
		src := make([]byte, n)
		rng.Read(src)
		enc := []byte(stdb32.StdEncoding.EncodeToString(src))
		if len(enc) > 0 {
			enc[rng.Intn(len(enc))] = "!@#$ \t019"[rng.Intn(9)]
		}
		gotB, gotErr := DecodeString(string(enc))
		wantB, wantErr := stdb32.StdEncoding.DecodeString(string(enc))
		if (gotErr == nil) != (wantErr == nil) {
			t.Fatalf("%q: err mismatch got=%v want=%v", enc, gotErr, wantErr)
		}
		if gotErr != nil {
			if gotErr.Error() != wantErr.Error() {
				t.Fatalf("%q: err offset mismatch got=%v want=%v", enc, gotErr, wantErr)
			}
			continue
		}
		if string(gotB) != string(wantB) {
			t.Fatalf("%q: decode mismatch", enc)
		}
	}
}

// TestDecodeSIMDDispatchVec exercises every branch of decodeSIMD: full<2 early
// return, the normal multi-block path, and the store-cap loop (via a tight dst)
// down to the maxBlocks==0 fallback.
func TestDecodeSIMDDispatchVec(t *testing.T) {
	rng := rand.New(rand.NewSource(31))

	if sd, dd := decodeSIMD(make([]byte, 64), make([]byte, 8)); sd != 0 || dd != 0 {
		t.Fatalf("full<2: want (0,0), got (%d,%d)", sd, dd)
	}

	for _, n := range []int{16, 24, 32, 40, 64, 128} {
		src := make([]byte, n)
		rng.Read(src)
		enc := []byte(stdb32.StdEncoding.EncodeToString(src))
		full := DecodedLen(len(enc))
		for cap := 0; cap <= full; cap++ {
			dst := make([]byte, cap)
			sd, dd := decodeSIMD(dst, enc)
			if dd != sd/8*5 || sd%8 != 0 {
				t.Fatalf("n=%d cap=%d: inconsistent (sd=%d,dd=%d)", n, cap, sd, dd)
			}
			if sd > 0 && dd-5+16 > cap {
				t.Fatalf("n=%d cap=%d: last store overruns dst (dd=%d)", n, cap, dd)
			}
		}
		dst := make([]byte, full)
		nn, err := Decode(dst, enc)
		if err != nil || string(dst[:nn]) != string(src) {
			t.Fatalf("n=%d exact: err=%v mismatch", n, err)
		}
	}
}
