import ctypes, numpy as np, platform
m = ctypes.CDLL("libm.so.6")
print(platform.node(), np.__version__)
feats = np._core._multiarray_umath.__cpu_features__
print("AVX512F", feats.get("AVX512F"), "AVX512_SKX", feats.get("AVX512_SKX"), "AVX2", feats.get("AVX2"))
rng = np.random.default_rng(1)
n = 2_000_000
p = rng.random(n, dtype=np.float32)
eps = 1e-6
def cmp(name, got, fn, restype, argtype, xs):
    f = getattr(m, fn); f.restype = restype; f.argtypes = [argtype]
    ref = np.array([f(v) for v in xs.tolist()], dtype=got.dtype)
    bad = np.flatnonzero(got.view(f"u{got.itemsize}") != ref.view(f"u{got.itemsize}"))
    print(f"{name:28s} dtype {got.dtype} mismatches {bad.size}/{len(xs)} ({bad.size/len(xs):.4%})")
x = 2 * p + eps; print("2*p+eps dtype", x.dtype)
cmp("log2 f32", np.log2(x), "log2f", ctypes.c_float, ctypes.c_float, x)
y = np.clip(1 - p, eps, 1.0); print("clip dtype", y.dtype)
cmp("log10 f32", np.log10(y), "log10f", ctypes.c_float, ctypes.c_float, y)
z = (rng.random(n, dtype=np.float32) * 20 - 20)
cmp("exp f32", np.exp(z), "expf", ctypes.c_float, ctypes.c_float, z)
d = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
cmp("log f64", np.log(d), "log", ctypes.c_double, ctypes.c_double, d)
cmp("log1p f64", np.log1p(-d), "log1p", ctypes.c_double, ctypes.c_double, -d)
e = -rng.random(n) * 5
f = m.pow; f.restype = ctypes.c_double; f.argtypes = [ctypes.c_double, ctypes.c_double]
got = np.power(10, e); ref = np.array([f(10.0, v) for v in e.tolist()])
print("pow(10,x) f64 mismatches", int((got.view('u8') != ref.view('u8')).sum()), "/", n)
# does the result depend on position/array length (SIMD body vs tail)?
a = np.log2(x); b = np.concatenate([np.log2(x[i:i+1]) for i in range(20000)])
print("log2 f32 full-vs-single-element mismatches", int((a[:20000].view('u4') != b.view('u4')).sum()))
a = np.exp(z); b = np.concatenate([np.exp(z[i:i+1]) for i in range(20000)])
print("exp f32 full-vs-single-element mismatches", int((a[:20000].view('u4') != b.view('u4')).sum()))
