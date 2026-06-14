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

// TestDecodeExhaustive hammers every input length 0..256 with many random
// buffers, round-tripping through our Decode and comparing byte-for-byte against
// encoding/base32. On an AVX2/VSX/vector host this drives the SIMD decode kernel
// (and its scalar tail) across all block alignments — the real-hardware gate when
// run from a precompiled binary (no fuzzing toolchain).
func TestDecodeExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	for n := 0; n <= 256; n++ {
		for trial := 0; trial < 64; trial++ {
			src := make([]byte, n)
			rng.Read(src)
			enc := base32.StdEncoding.EncodeToString(src)
			dst := make([]byte, DecodedLen(len(enc)))
			nn, err := Decode(dst, []byte(enc))
			if err != nil {
				t.Fatalf("n=%d trial=%d: decode err: %v", n, trial, err)
			}
			if string(dst[:nn]) != string(src) {
				t.Fatalf("n=%d trial=%d: round-trip mismatch", n, trial)
			}
		}
	}
}

// TestDecodeErrorExhaustive corrupts random alphabet strings (and feeds raw
// random bytes) and requires the error/offset to match encoding/base32 exactly,
// exercising the kernel's per-block validity bail and the offset-shift in Decode.
func TestDecodeErrorExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(103))
	for trial := 0; trial < 200000; trial++ {
		var enc []byte
		switch trial % 3 {
		case 0:
			n := rng.Intn(60)
			src := make([]byte, n)
			rng.Read(src)
			enc = []byte(base32.StdEncoding.EncodeToString(src))
			if len(enc) > 0 {
				enc[rng.Intn(len(enc))] = "!@#$ \t01239xyz="[rng.Intn(15)]
			}
		case 1:
			n := rng.Intn(60)
			enc = make([]byte, n)
			cs := "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567=!@ 019"
			for i := range enc {
				enc[i] = cs[rng.Intn(len(cs))]
			}
		default:
			n := rng.Intn(60)
			enc = make([]byte, n)
			rng.Read(enc)
		}
		gotB, gotErr := DecodeString(string(enc))
		wantB, wantErr := base32.StdEncoding.DecodeString(string(enc))
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
			t.Fatalf("%q: decode mismatch got=%x want=%x", enc, gotB, wantB)
		}
	}
}

func BenchmarkDecode(b *testing.B) {
	src := benchData()
	enc := []byte(base32.StdEncoding.EncodeToString(src))
	dst := make([]byte, DecodedLen(len(enc)))
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Decode(dst, enc)
	}
}

func BenchmarkDecodeStdlib(b *testing.B) {
	src := benchData()
	enc := []byte(base32.StdEncoding.EncodeToString(src))
	dst := make([]byte, DecodedLen(len(enc)))
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		base32.StdEncoding.Decode(dst, enc)
	}
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
