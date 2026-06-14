//go:build amd64

package base32

import (
	stdb32 "encoding/base32"
	"math/rand"
	"testing"
)

// decodeForce runs a chosen decode kernel (SSE or AVX2) over its valid-block
// prefix then delegates the remainder to encoding/base32, so each path can be
// exercised independently of CPU dispatch.
func decodeForce(dst, src []byte, avx2 bool) (int, error) {
	full := len(src) / 8
	var sd, dd int
	if full >= 2 {
		maxBlocks := full - 1
		var blocks int
		if avx2 {
			blocks = decodeBlocksAVX2(dst, src, maxBlocks)
		} else {
			blocks = decodeBlocksSSE(dst, src, maxBlocks)
		}
		sd, dd = blocks*8, blocks*5
	}
	n, err := stdb32.StdEncoding.Decode(dst[dd:], src[sd:])
	if err != nil {
		if ce, ok := err.(stdb32.CorruptInputError); ok {
			return dd + n, stdb32.CorruptInputError(int64(ce) + int64(sd))
		}
		return dd + n, err
	}
	return dd + n, nil
}

func TestForceDecodeSSE(t *testing.T)  { testForceDecode(t, false) }
func TestForceDecodeAVX2(t *testing.T) { testForceDecode(t, true) }

func testForceDecode(t *testing.T, avx2 bool) {
	rng := rand.New(rand.NewSource(13))
	for _, n := range []int{0, 1, 4, 5, 10, 16, 20, 21, 25, 30, 31, 32, 33, 100, 1000, 4096} {
		src := make([]byte, n)
		rng.Read(src)
		enc := []byte(stdb32.StdEncoding.EncodeToString(src))
		dst := make([]byte, DecodedLen(len(enc)))
		nn, err := decodeForce(dst, enc, avx2)
		if err != nil {
			t.Fatalf("avx2=%v n=%d: decode err: %v", avx2, n, err)
		}
		if string(dst[:nn]) != string(src) {
			t.Fatalf("avx2=%v n=%d: round-trip mismatch\n got=%x\nwant=%x", avx2, n, dst[:nn], src)
		}
	}
}

// TestForceDecodeInvalid drives the kernel's per-block validity bail (an invalid
// char inside an otherwise-full block stops the kernel and the remainder, with
// the correct error offset, comes from encoding/base32). Errors must be
// byte-and-offset identical to the stdlib for both kernels.
func TestForceDecodeInvalid(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for _, avx2 := range []bool{false, true} {
		for trial := 0; trial < 2000; trial++ {
			n := rng.Intn(50)
			src := make([]byte, n)
			rng.Read(src)
			enc := []byte(stdb32.StdEncoding.EncodeToString(src))
			if len(enc) > 0 {
				// corrupt a random char to a non-alphabet byte
				enc[rng.Intn(len(enc))] = "!@#$ \t01"[rng.Intn(8)]
			}
			dst := make([]byte, DecodedLen(len(enc)))
			gotN, gotErr := decodeForce(dst, enc, avx2)
			wdst := make([]byte, DecodedLen(len(enc)))
			wantN, wantErr := stdb32.StdEncoding.Decode(wdst, enc)
			if (gotErr == nil) != (wantErr == nil) {
				t.Fatalf("avx2=%v %q: err mismatch got=%v want=%v", avx2, enc, gotErr, wantErr)
			}
			if gotErr != nil {
				if gotErr.Error() != wantErr.Error() {
					t.Fatalf("avx2=%v %q: err offset mismatch got=%v want=%v", avx2, enc, gotErr, wantErr)
				}
				continue
			}
			if gotN != wantN || string(dst[:gotN]) != string(wdst[:wantN]) {
				t.Fatalf("avx2=%v %q: decode mismatch", avx2, enc)
			}
		}
	}
}

// TestDecodeSIMDDispatch drives every branch of the amd64 decodeSIMD dispatcher
// through the public Decode API: the AVX2 path, the SSE path (hasAVX2 forced
// low), and the full<2 early return. CI runs on a native AVX2 box where hasAVX2
// is otherwise always true (restored via defer).
func TestDecodeSIMDDispatch(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	check := func(n int) {
		src := make([]byte, n)
		rng.Read(src)
		enc := stdb32.StdEncoding.EncodeToString(src)
		got, err := DecodeString(enc)
		if err != nil {
			t.Fatalf("n=%d hasAVX2=%v: %v", n, hasAVX2, err)
		}
		if string(got) != string(src) {
			t.Fatalf("n=%d hasAVX2=%v: mismatch", n, hasAVX2)
		}
	}
	ns := []int{0, 1, 5, 10, 11, 15, 16, 20, 21, 100, 1000}
	for _, n := range ns {
		check(n)
	}
	saved := hasAVX2
	defer func() { hasAVX2 = saved }()
	hasAVX2 = false
	for _, n := range ns {
		check(n)
	}
}
