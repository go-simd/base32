package base32

import (
	"encoding/base32"
	"math/rand"
	"testing"
)

func TestEncode(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 15, 16, 20, 21, 25, 31, 32, 33, 1000, 1024, 64 * 1024} {
		src := make([]byte, n)
		rng.Read(src)
		got, want := EncodeToString(src), base32.StdEncoding.EncodeToString(src)
		if got != want {
			t.Fatalf("n=%d:\n got=%q\nwant=%q", n, got, want)
		}
		// Round-trip through our Decode (scalar/stdlib) back to src.
		dec, err := DecodeString(got)
		if err != nil {
			t.Fatalf("n=%d decode: %v", n, err)
		}
		if string(dec) != string(src) {
			t.Fatalf("n=%d round-trip mismatch", n)
		}
	}
}

func TestDecode(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{0, 1, 2, 4, 5, 10, 16, 21, 100, 1000} {
		src := make([]byte, n)
		rng.Read(src)
		enc := EncodeToString(src)

		// DecodedLen must be large enough for the decode buffer.
		if got := DecodedLen(len(enc)); got < n {
			t.Fatalf("n=%d: DecodedLen=%d < %d", n, got, n)
		}
		// Decode into a caller-supplied buffer.
		dst := make([]byte, DecodedLen(len(enc)))
		nn, err := Decode(dst, []byte(enc))
		if err != nil {
			t.Fatalf("n=%d: Decode: %v", n, err)
		}
		if string(dst[:nn]) != string(src) {
			t.Fatalf("n=%d: Decode round-trip mismatch", n)
		}
	}
}

func TestDecodeError(t *testing.T) {
	// Invalid base32 must surface the stdlib error through both wrappers.
	if _, err := DecodeString("1!!!!!!!"); err == nil {
		t.Fatal("DecodeString: want error on invalid input, got nil")
	}
	dst := make([]byte, 8)
	if _, err := Decode(dst, []byte("1!!!!!!!")); err == nil {
		t.Fatal("Decode: want error on invalid input, got nil")
	}
}

func FuzzEncode(f *testing.F) {
	f.Add([]byte("hello world"))
	f.Add([]byte(""))
	f.Add([]byte("\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"))
	f.Fuzz(func(t *testing.T, src []byte) {
		if got, want := EncodeToString(src), base32.StdEncoding.EncodeToString(src); got != want {
			t.Fatalf("got=%q want=%q", got, want)
		}
	})
}

func FuzzDecode(f *testing.F) {
	f.Add("ME======")
	f.Add("MFRGGZDFMZTWQ2LK")
	f.Add("")
	f.Fuzz(func(t *testing.T, s string) {
		gotB, gotErr := DecodeString(s)
		wantB, wantErr := base32.StdEncoding.DecodeString(s)
		if (gotErr == nil) != (wantErr == nil) {
			t.Fatalf("err mismatch for %q: got=%v want=%v", s, gotErr, wantErr)
		}
		if gotErr == nil && string(gotB) != string(wantB) {
			t.Fatalf("decode mismatch for %q: got=%x want=%x", s, gotB, wantB)
		}
	})
}

func benchData() []byte { b := make([]byte, 1<<20); rand.New(rand.NewSource(2)).Read(b); return b }

func BenchmarkEncode(b *testing.B) {
	src := benchData()
	dst := make([]byte, EncodedLen(len(src)))
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(dst, src)
	}
}

func BenchmarkEncodeStdlib(b *testing.B) {
	src := benchData()
	dst := make([]byte, EncodedLen(len(src)))
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		base32.StdEncoding.Encode(dst, src)
	}
}

// TestEncodeExhaustive hammers every length 0..256 with many random buffers,
// comparing byte-for-byte against encoding/base32. On an AVX2 host this drives
// the AVX2 kernel (and its SSE/scalar tails) across all alignments — used as the
// real-hardware gate when run from a precompiled binary (no fuzzing toolchain).
func TestEncodeExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for n := 0; n <= 256; n++ {
		for trial := 0; trial < 64; trial++ {
			src := make([]byte, n)
			rng.Read(src)
			if got, want := EncodeToString(src), base32.StdEncoding.EncodeToString(src); got != want {
				t.Fatalf("n=%d trial=%d:\n got=%q\nwant=%q", n, trial, got, want)
			}
		}
	}
}
