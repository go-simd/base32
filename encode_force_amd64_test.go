//go:build amd64

package base32

import (
	stdb32 "encoding/base32"
	"math/rand"
	"testing"
)

// encodeForce runs a chosen kernel (SSE or AVX2) over its in-bounds prefix then
// delegates the tail to encoding/base32, so each path can be exercised
// independently of CPU dispatch.
func encodeForce(dst, src []byte, avx2 bool) {
	n := len(src)
	if avx2 && n >= 21 {
		g := (n-21)/10 + 1
		encodeBlocksAVX2(dst, src, g)
		stdb32.StdEncoding.Encode(dst[g*16:], src[g*10:])
		return
	}
	if n >= 16 {
		g := (n-16)/5 + 1
		encodeBlocksSSE(dst, src, g)
		stdb32.StdEncoding.Encode(dst[g*8:], src[g*5:])
		return
	}
	stdb32.StdEncoding.Encode(dst, src)
}

func encodeForceToString(src []byte, avx2 bool) string {
	dst := make([]byte, EncodedLen(len(src)))
	encodeForce(dst, src, avx2)
	return string(dst)
}

// TestForceSSE and TestForceAVX2 verify each kernel byte-for-byte vs the stdlib
// regardless of whether the host CPU advertises the feature (the kernel still
// executes; on a CPU lacking AVX2 the OS would fault, but CI runs it on real
// AVX2 and AVX2 is gated there).
func TestForceSSE(t *testing.T)  { testForce(t, false) }
func TestForceAVX2(t *testing.T) { testForce(t, true) }

func testForce(t *testing.T, avx2 bool) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{0, 1, 4, 5, 9, 10, 15, 16, 20, 21, 25, 30, 31, 32, 33, 100, 1000, 4096} {
		src := make([]byte, n)
		rng.Read(src)
		got := encodeForceToString(src, avx2)
		want := stdb32.StdEncoding.EncodeToString(src)
		if got != want {
			t.Fatalf("avx2=%v n=%d:\n got=%q\nwant=%q", avx2, n, got, want)
		}
	}
}

func benchForce(b *testing.B, avx2 bool) {
	src := make([]byte, 1<<20)
	rand.New(rand.NewSource(2)).Read(src)
	dst := make([]byte, EncodedLen(len(src)))
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encodeForce(dst, src, avx2)
	}
}

func BenchmarkEncodeForceSSE(b *testing.B)  { benchForce(b, false) }
func BenchmarkEncodeForceAVX2(b *testing.B) { benchForce(b, true) }
