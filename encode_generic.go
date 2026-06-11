//go:build !amd64

package base32

// encodeSIMD has no SIMD kernel on this arch; the whole input goes to the
// standard library.
func encodeSIMD(dst, src []byte) (srcDone, dstDone int) { return 0, 0 }
