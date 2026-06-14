//go:build !amd64 && !ppc64le && !s390x && !(arm64 && go1.27)

package base32

// encodeSIMD has no SIMD kernel on this arch; the whole input goes to the
// standard library.
func encodeSIMD(dst, src []byte) (srcDone, dstDone int) { return 0, 0 }

// decodeSIMD has no SIMD kernel on this arch; the whole input goes to the
// standard library.
func decodeSIMD(dst, src []byte) (srcDone, dstDone int) { return 0, 0 }
