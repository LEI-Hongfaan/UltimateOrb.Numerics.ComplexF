// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics.CodeAnalysis;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

[assembly: CLSCompliant(true)]

namespace UltimateOrb.Numerics {

    /// <summary>
    /// A complex number z is a number of the form z = x + yi, where x and y
    /// are real numbers, and i is the imaginary unit, with the property i2= -1.
    /// </summary>
    [Serializable]
    // [TypeForwardedFrom("System.Numerics, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089")]
    public readonly struct ComplexF
        : IEquatable<ComplexF>,
          IFormattable,
          INumberBase<ComplexF>,
          ISignedNumber<ComplexF>,
          IUtf8SpanFormattable {
        private const NumberStyles DefaultNumberStyle = NumberStyles.Float | NumberStyles.AllowThousands;

        private const NumberStyles InvalidNumberStyles = ~(NumberStyles.AllowLeadingWhite | NumberStyles.AllowTrailingWhite
                                                         | NumberStyles.AllowLeadingSign | NumberStyles.AllowTrailingSign
                                                         | NumberStyles.AllowParentheses | NumberStyles.AllowDecimalPoint
                                                         | NumberStyles.AllowThousands | NumberStyles.AllowExponent
                                                         | NumberStyles.AllowCurrencySymbol | NumberStyles.AllowHexSpecifier);

        public static readonly ComplexF Zero = new ComplexF(0.0F, 0.0F);
        public static readonly ComplexF One = new ComplexF(1.0F, 0.0F);
        public static readonly ComplexF ImaginaryOne = new ComplexF(0.0F, 1.0F);
        public static readonly ComplexF NaN = new ComplexF(float.NaN, float.NaN);
        public static readonly ComplexF Infinity = new ComplexF(float.PositiveInfinity, float.PositiveInfinity);

        private const float InverseOfLog10 = 0.43429448190325F; // 1 / Log(10)

        // This is the largest x for which (Hypot(x,x) + x) will not overflow. It is used for branching inside Sqrt.
        private static readonly float s_sqrtRescaleThreshold = (float)(float.MaxValue / (Math.Sqrt(2.0) + 1.0));

        // This is the largest x for which 2 x^2 will not overflow. It is used for branching inside Asin and Acos.
        private static readonly float s_asinOverflowThreshold = (float)(Math.Sqrt(float.MaxValue) / 2.0);

        // This value is used inside Asin and Acos.
        private static readonly float s_log2 = MathF.Log(2.0F);

        // Do not rename, these fields are needed for binary serialization
        private readonly float m_real; // Do not rename (binary serialization)
        private readonly float m_imaginary; // Do not rename (binary serialization)

        public ComplexF(float real, float imaginary) {
            m_real = real;
            m_imaginary = imaginary;
        }

        public float Real { get { return m_real; } }
        public float Imaginary { get { return m_imaginary; } }

        public float Magnitude { get { return Abs(this); } }
        public float Phase { get { return MathF.Atan2(m_imaginary, m_real); } }

        public static ComplexF FromPolarCoordinates(float magnitude, float phase) {
            var (sin, cos) = Math.SinCos(phase);
            return new ComplexF((float)(magnitude * cos), (float)(magnitude * sin));
        }

        public static ComplexF Negate(ComplexF value) {
            return -value;
        }

        public static ComplexF Add(ComplexF left, ComplexF right) {
            return left + right;
        }

        public static ComplexF Add(ComplexF left, float right) {
            return left + right;
        }

        public static ComplexF Add(float left, ComplexF right) {
            return left + right;
        }

        public static ComplexF Subtract(ComplexF left, ComplexF right) {
            return left - right;
        }

        public static ComplexF Subtract(ComplexF left, float right) {
            return left - right;
        }

        public static ComplexF Subtract(float left, ComplexF right) {
            return left - right;
        }

        public static ComplexF Multiply(ComplexF left, ComplexF right) {
            return left * right;
        }

        public static ComplexF Multiply(ComplexF left, float right) {
            return left * right;
        }

        public static ComplexF Multiply(float left, ComplexF right) {
            return left * right;
        }

        public static ComplexF Divide(ComplexF dividend, ComplexF divisor) {
            return dividend / divisor;
        }

        public static ComplexF Divide(ComplexF dividend, float divisor) {
            return dividend / divisor;
        }

        public static ComplexF Divide(float dividend, ComplexF divisor) {
            return dividend / divisor;
        }

        public static ComplexF operator -(ComplexF value)  /* Unary negation of a complex number */
        {
            return new ComplexF(-value.m_real, -value.m_imaginary);
        }

        public static ComplexF operator +(ComplexF left, ComplexF right) {
            return new ComplexF(left.m_real + right.m_real, left.m_imaginary + right.m_imaginary);
        }

        public static ComplexF operator +(ComplexF left, float right) {
            return new ComplexF(left.m_real + right, left.m_imaginary);
        }

        public static ComplexF operator +(float left, ComplexF right) {
            return new ComplexF(left + right.m_real, right.m_imaginary);
        }

        public static ComplexF operator -(ComplexF left, ComplexF right) {
            return new ComplexF(left.m_real - right.m_real, left.m_imaginary - right.m_imaginary);
        }

        public static ComplexF operator -(ComplexF left, float right) {
            return new ComplexF(left.m_real - right, left.m_imaginary);
        }

        public static ComplexF operator -(float left, ComplexF right) {
            return new ComplexF(left - right.m_real, -right.m_imaginary);
        }

        public static ComplexF operator *(ComplexF left, ComplexF right) {
            // Multiplication:  (a + bi)(c + di) = (ac -bd) + (bc + ad)i
            float result_realpart = (left.m_real * right.m_real) - (left.m_imaginary * right.m_imaginary);
            float result_imaginarypart = (left.m_imaginary * right.m_real) + (left.m_real * right.m_imaginary);
            return new ComplexF(result_realpart, result_imaginarypart);
        }

        public static ComplexF operator *(ComplexF left, float right) {
            if (!float.IsFinite(left.m_real)) {
                if (!float.IsFinite(left.m_imaginary)) {
                    return new ComplexF(float.NaN, float.NaN);
                }

                return new ComplexF(left.m_real * right, float.NaN);
            }

            if (!float.IsFinite(left.m_imaginary)) {
                return new ComplexF(float.NaN, left.m_imaginary * right);
            }

            return new ComplexF(left.m_real * right, left.m_imaginary * right);
        }

        public static ComplexF operator *(float left, ComplexF right) {
            if (!float.IsFinite(right.m_real)) {
                if (!float.IsFinite(right.m_imaginary)) {
                    return new ComplexF(float.NaN, float.NaN);
                }

                return new ComplexF(left * right.m_real, float.NaN);
            }

            if (!float.IsFinite(right.m_imaginary)) {
                return new ComplexF(float.NaN, left * right.m_imaginary);
            }

            return new ComplexF(left * right.m_real, left * right.m_imaginary);
        }

        public static ComplexF operator /(ComplexF left, ComplexF right) {
            // Division : Smith's formula.
            float a = left.m_real;
            float b = left.m_imaginary;
            float c = right.m_real;
            float d = right.m_imaginary;

            // Computing c * c + d * d will overflow even in cases where the actual result of the division does not overflow.
            if (MathF.Abs(d) < MathF.Abs(c)) {
                float doc = d / c;
                return new ComplexF((a + b * doc) / (c + d * doc), (b - a * doc) / (c + d * doc));
            } else {
                float cod = c / d;
                return new ComplexF((b + a * cod) / (d + c * cod), (-a + b * cod) / (d + c * cod));
            }
        }

        public static ComplexF operator /(ComplexF left, float right) {
            // IEEE prohibit optimizations which are value changing
            // so we make sure that behaviour for the simplified version exactly match
            // full version.
            if (right == 0F) {
                return new ComplexF(float.NaN, float.NaN);
            }

            if (!float.IsFinite(left.m_real)) {
                if (!float.IsFinite(left.m_imaginary)) {
                    return new ComplexF(float.NaN, float.NaN);
                }

                return new ComplexF(left.m_real / right, float.NaN);
            }

            if (!float.IsFinite(left.m_imaginary)) {
                return new ComplexF(float.NaN, left.m_imaginary / right);
            }

            // Here the actual optimized version of code.
            return new ComplexF(left.m_real / right, left.m_imaginary / right);
        }

        public static ComplexF operator /(float left, ComplexF right) {
            // Division : Smith's formula.
            float a = left;
            float c = right.m_real;
            float d = right.m_imaginary;

            // Computing c * c + d * d will overflow even in cases where the actual result of the division does not overflow.
            if (MathF.Abs(d) < MathF.Abs(c)) {
                float doc = d / c;
                return new ComplexF(a / (c + d * doc), (-a * doc) / (c + d * doc));
            } else {
                float cod = c / d;
                return new ComplexF(a * cod / (d + c * cod), -a / (d + c * cod));
            }
        }

        public static float Abs(ComplexF value) {
            return Hypot(value.m_real, value.m_imaginary);
        }

        private static float Hypot(float a, float b) {
            // Using
            //   sqrt(a^2 + b^2) = |a| * sqrt(1 + (b/a)^2)
            // we can factor out the larger component to dodge overflow even when a * a would overflow.

            a = MathF.Abs(a);
            b = MathF.Abs(b);

            float small, large;
            if (a < b) {
                small = a;
                large = b;
            } else {
                small = b;
                large = a;
            }

            if (small == 0.0F) {
                return (large);
            } else if (float.IsPositiveInfinity(large) && !float.IsNaN(small)) {
                // The NaN test is necessary so we don't return +inf when small=NaN and large=+inf.
                // NaN in any other place returns NaN without any special handling.
                return (float.PositiveInfinity);
            } else {
                float ratio = small / large;
                return (large * MathF.Sqrt(1.0F + ratio * ratio));
            }

        }

        private static double Hypot(double a, double b) {
            // Using
            //   sqrt(a^2 + b^2) = |a| * sqrt(1 + (b/a)^2)
            // we can factor out the larger component to dodge overflow even when a * a would overflow.

            a = Math.Abs(a);
            b = Math.Abs(b);

            double small, large;
            if (a < b) {
                small = a;
                large = b;
            } else {
                small = b;
                large = a;
            }

            if (small == 0.0) {
                return (large);
            } else if (double.IsPositiveInfinity(large) && !double.IsNaN(small)) {
                // The NaN test is necessary so we don't return +inf when small=NaN and large=+inf.
                // NaN in any other place returns NaN without any special handling.
                return (double.PositiveInfinity);
            } else {
                double ratio = small / large;
                return (large * Math.Sqrt(1.0 + ratio * ratio));
            }

        }

        private static float Log1P(float x) {
            // Compute log(1 + x) without loss of accuracy when x is small.

            // Our only use case so far is for positive values, so this isn't coded to handle negative values.
            Debug.Assert((x >= 0.0F) || float.IsNaN(x));

            float xp1 = 1.0F + x;
            if (xp1 == 1.0F) {
                return x;
            } else if (x < 0.75F) {
                // This is accurate to within 5 ulp with any floating-point system that uses a guard digit,
                // as proven in Theorem 4 of "What Every Computer Scientist Should Know About Floating-Point
                // Arithmetic" (https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
                return x * MathF.Log(xp1) / (xp1 - 1.0F);
            } else {
                return MathF.Log(xp1);
            }

        }

        private static double Log1P(double x) {
            // Compute log(1 + x) without loss of accuracy when x is small.

            // Our only use case so far is for positive values, so this isn't coded to handle negative values.
            Debug.Assert((x >= 0.0) || double.IsNaN(x));

            double xp1 = 1.0 + x;
            if (xp1 == 1.0) {
                return x;
            } else if (x < 0.75) {
                // This is accurate to within 5 ulp with any floating-point system that uses a guard digit,
                // as proven in Theorem 4 of "What Every Computer Scientist Should Know About Floating-Point
                // Arithmetic" (https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
                return x * Math.Log(xp1) / (xp1 - 1.0);
            } else {
                return Math.Log(xp1);
            }

        }

        public static ComplexF Conjugate(ComplexF value) {
            // Conjugate of a ComplexF number: the conjugate of x+i*y is x-i*y
            return new ComplexF(value.m_real, -value.m_imaginary);
        }

        public static ComplexF Reciprocal(ComplexF value) {
            // Reciprocal of a ComplexF number : the reciprocal of x+i*y is 1/(x+i*y)
            if (value.m_real == 0F && value.m_imaginary == 0F) {
                return Zero;
            }
            return One / value;
        }

        public static bool operator ==(ComplexF left, ComplexF right) {
            return left.m_real == right.m_real && left.m_imaginary == right.m_imaginary;
        }

        public static bool operator !=(ComplexF left, ComplexF right) {
            return left.m_real != right.m_real || left.m_imaginary != right.m_imaginary;
        }

        public override bool Equals([NotNullWhen(true)] object? obj) {
            return obj is ComplexF other && Equals(other);
        }

        public bool Equals(ComplexF value) {
            return m_real.Equals(value.m_real) && m_imaginary.Equals(value.m_imaginary);
        }

        public override int GetHashCode() => HashCode.Combine(m_real, m_imaginary);

        public override string ToString() => ToString(null, null);

        public string ToString([StringSyntax(StringSyntaxAttribute.NumericFormat)] string? format) => ToString(format, null);

        public string ToString(IFormatProvider? provider) => ToString(null, provider);

        public string ToString([StringSyntax(StringSyntaxAttribute.NumericFormat)] string? format, IFormatProvider? provider) {
            // $"<{m_real.ToString(format, provider)}; {m_imaginary.ToString(format, provider)}>";
            var handler = new DefaultInterpolatedStringHandler(4, 2, provider, stackalloc char[512]);
            handler.AppendLiteral("<");
            handler.AppendFormatted(m_real, format);
            handler.AppendLiteral("; ");
            handler.AppendFormatted(m_imaginary, format);
            handler.AppendLiteral(">");
            return handler.ToStringAndClear();
        }

        public static ComplexF Sin(ComplexF value) {
            // We need both sinh and cosh of imaginary part. To avoid multiple calls to Math.Exp with the same value,
            // we compute them both here from a single call to Math.Exp.
            float p = MathF.Exp(value.m_imaginary);
            float q = 1.0F / p;
            float sinh = (p - q) * 0.5F;
            float cosh = (p + q) * 0.5F;
            (float sin, float cos) = MathF.SinCos(value.m_real);
            return new ComplexF(sin * cosh, cos * sinh);
            // There is a known limitation with this algorithm: inputs that cause sinh and cosh to overflow, but for
            // which sin or cos are small enough that sin * cosh or cos * sinh are still representable, nonetheless
            // produce overflow. For example, Sin((0.01, 711.0)) should produce (~3.0E306, PositiveInfinity), but
            // instead produces (PositiveInfinity, PositiveInfinity).
        }

        public static ComplexF Sinh(ComplexF value) {
            // Use sinh(z) = -i sin(iz) to compute via sin(z).
            ComplexF sin = Sin(new ComplexF(-value.m_imaginary, value.m_real));
            return new ComplexF(sin.m_imaginary, -sin.m_real);
        }

        public static ComplexF Asin(ComplexF value) {
            float b, bPrime, v;
            Asin_Internal(MathF.Abs(value.Real), MathF.Abs(value.Imaginary), out b, out bPrime, out v);

            float u;
            if (bPrime < 0.0F) {
                u = MathF.Asin(b);
            } else {
                u = MathF.Atan(bPrime);
            }

            if (value.Real < 0.0F) u = -u;
            if (value.Imaginary < 0.0F) v = -v;

            return new ComplexF(u, v);
        }

        public static ComplexF Cos(ComplexF value) {
            float p = MathF.Exp(value.m_imaginary);
            float q = 1.0F / p;
            float nsinh = (q - p) * 0.5F;
            float cosh = (q + p) * 0.5F;
            (float sin, float cos) = MathF.SinCos(value.m_real);
            return new ComplexF(cos * cosh, sin * nsinh);
        }

        public static ComplexF Cosh(ComplexF value) {
            // Use cosh(z) = cos(iz) to compute via cos(z).
            return Cos(new ComplexF(-value.m_imaginary, value.m_real));
        }

        public static ComplexF Acos(ComplexF value) {
            float b, bPrime, v;
            Asin_Internal(MathF.Abs(value.Real), MathF.Abs(value.Imaginary), out b, out bPrime, out v);

            float u;
            if (bPrime < 0.0F) {
                u = MathF.Acos(b);
            } else {
                u = MathF.Atan(1.0F / bPrime);
            }

            if (value.Real < 0.0F) u = MathF.PI - u;
            if (value.Imaginary > 0.0F) v = -v;

            return new ComplexF(u, v);
        }

        public static ComplexF Tan(ComplexF value) {
            // tan z = sin z / cos z, but to avoid unnecessary repeated trig computations, use
            //   tan z = (sin(2x) + i sinh(2y)) / (cos(2x) + cosh(2y))
            // (see Abramowitz & Stegun 4.3.57 or derive by hand), and compute trig functions here.

            // This approach does not work for |y| > ~355, because sinh(2y) and cosh(2y) overflow,
            // even though their ratio does not. In that case, divide through by cosh to get:
            //   tan z = (sin(2x) / cosh(2y) + i \tanh(2y)) / (1 + cos(2x) / cosh(2y))
            // which correctly computes the (tiny) real part and the (normal-sized) imaginary part.

            float x2 = 2.0F * value.m_real;
            float y2 = 2.0F * value.m_imaginary;
            float p = MathF.Exp(y2);
            float q = 1.0F / p;
            float cosh = (p + q) * 0.5F;
            (float sin, float cos) = MathF.SinCos(x2);
            if (Math.Abs(value.m_imaginary) <= 4.0F) {
                float sinh = (p - q) * 0.5F;
                float D = cos + cosh;
                return new ComplexF(sin / D, sinh / D);
            } else {
                float D = 1.0F + cos / cosh;
                return new ComplexF(sin / cosh / D, MathF.Tanh(y2) / D);
            }
        }

        public static ComplexF Tanh(ComplexF value) {
            // Use tanh(z) = -i tan(iz) to compute via tan(z).
            ComplexF tan = Tan(new ComplexF(-value.m_imaginary, value.m_real));
            return new ComplexF(tan.m_imaginary, -tan.m_real);
        }

        public static ComplexF Atan(ComplexF value) {
            ComplexF two = new ComplexF(2.0F, 0.0F);
            return (ImaginaryOne / two) * (Log(One - ImaginaryOne * value) - Log(One + ImaginaryOne * value));
        }

        private static void Asin_Internal(double x, double y, out double b, out double bPrime, out double v) {

            // This method for the inverse complex sine (and cosine) is described in Hull, Fairgrieve,
            // and Tang, "Implementing the ComplexF Arcsine and Arccosine Functions Using Exception Handling",
            // ACM Transactions on Mathematical Software (1997)
            // (https://www.researchgate.net/profile/Ping_Tang3/publication/220493330_Implementing_the_Complex_Arcsine_and_Arccosine_Functions_Using_Exception_Handling/links/55b244b208ae9289a085245d.pdf)

            // First, the basics: start with sin(w) = (e^{iw} - e^{-iw}) / (2i) = z. Here z is the input
            // and w is the output. To solve for w, define t = e^{i w} and multiply through by t to
            // get the quadratic equation t^2 - 2 i z t - 1 = 0. The solution is t = i z + sqrt(1 - z^2), so
            //   w = arcsin(z) = - i log( i z + sqrt(1 - z^2) )
            // Decompose z = x + i y, multiply out i z + sqrt(1 - z^2), use log(s) = |s| + i arg(s), and do a
            // bunch of algebra to get the components of w = arcsin(z) = u + i v
            //   u = arcsin(beta)  v = sign(y) log(alpha + sqrt(alpha^2 - 1))
            // where
            //   alpha = (rho + sigma) / 2      beta = (rho - sigma) / 2
            //   rho = sqrt((x + 1)^2 + y^2)    sigma = sqrt((x - 1)^2 + y^2)
            // These formulas appear in DLMF section 4.23. (http://dlmf.nist.gov/4.23), along with the analogous
            //   arccos(w) = arccos(beta) - i sign(y) log(alpha + sqrt(alpha^2 - 1))
            // So alpha and beta together give us arcsin(w) and arccos(w).

            // As written, alpha is not susceptible to cancelation errors, but beta is. To avoid cancelation, note
            //   beta = (rho^2 - sigma^2) / (rho + sigma) / 2 = (2 x) / (rho + sigma) = x / alpha
            // which is not subject to cancelation. Note alpha >= 1 and |beta| <= 1.

            // For alpha ~ 1, the argument of the log is near unity, so we compute (alpha - 1) instead,
            // write the argument as 1 + (alpha - 1) + sqrt((alpha - 1)(alpha + 1)), and use the log1p function
            // to compute the log without loss of accuracy.
            // For beta ~ 1, arccos does not accurately resolve small angles, so we compute the tangent of the angle
            // instead.
            // Hull, Fairgrieve, and Tang derive formulas for (alpha - 1) and beta' = tan(u) that do not suffer
            // from cancelation in these cases.

            // For simplicity, we assume all positive inputs and return all positive outputs. The caller should
            // assign signs appropriate to the desired cut conventions. We return v directly since its magnitude
            // is the same for both arcsin and arccos. Instead of u, we usually return beta and sometimes beta'.
            // If beta' is not computed, it is set to -1; if it is computed, it should be used instead of beta
            // to determine u. Compute u = arcsin(beta) or u = arctan(beta') for arcsin, u = arccos(beta)
            // or arctan(1/beta') for arccos.

            Debug.Assert((x >= 0.0) || double.IsNaN(x));
            Debug.Assert((y >= 0.0) || double.IsNaN(y));

            // For x or y large enough to overflow alpha^2, we can simplify our formulas and avoid overflow.
            if ((x > s_asinOverflowThreshold) || (y > s_asinOverflowThreshold)) {
                b = -1.0;
                bPrime = x / y;

                double small, big;
                if (x < y) {
                    small = x;
                    big = y;
                } else {
                    small = y;
                    big = x;
                }
                double ratio = small / big;
                v = s_log2 + Math.Log(big) + 0.5 * Log1P(ratio * ratio);
            } else {
                double r = Hypot((x + 1.0), y);
                double s = Hypot((x - 1.0), y);

                double a = (r + s) * 0.5;
                b = x / a;

                if (b > 0.75) {
                    if (x <= 1.0) {
                        double amx = (y * y / (r + (x + 1.0)) + (s + (1.0 - x))) * 0.5;
                        bPrime = x / Math.Sqrt((a + x) * amx);
                    } else {
                        // In this case, amx ~ y^2. Since we take the square root of amx, we should
                        // pull y out from under the square root so we don't lose its contribution
                        // when y^2 underflows.
                        double t = (1.0 / (r + (x + 1.0)) + 1.0 / (s + (x - 1.0))) * 0.5;
                        bPrime = x / y / Math.Sqrt((a + x) * t);
                    }
                } else {
                    bPrime = -1.0;
                }

                if (a < 1.5) {
                    if (x < 1.0) {
                        // This is another case where our expression is proportional to y^2 and
                        // we take its square root, so again we pull out a factor of y from
                        // under the square root.
                        double t = (1.0 / (r + (x + 1.0)) + 1.0 / (s + (1.0 - x))) * 0.5;
                        double am1 = y * y * t;
                        v = Log1P(am1 + y * Math.Sqrt(t * (a + 1.0)));
                    } else {
                        double am1 = (y * y / (r + (x + 1.0)) + (s + (x - 1.0))) * 0.5;
                        v = Log1P(am1 + Math.Sqrt(am1 * (a + 1.0)));
                    }
                } else {
                    // Because of the test above, we can be sure that a * a will not overflow.
                    v = Math.Log(a + Math.Sqrt((a - 1.0) * (a + 1.0)));
                }
            }
        }

        private static void Asin_Internal(float x, float y, out float b, out float bPrime, out float v) {
            double b_, bPrime_, v_;
            Asin_Internal(x, y, out b_, out bPrime_, out v_);
            b = (float)b_;
            bPrime = (float)bPrime_;
            v = (float)v_;
        }

        public static bool IsFinite(ComplexF value) => float.IsFinite(value.m_real) && float.IsFinite(value.m_imaginary);

        public static bool IsInfinity(ComplexF value) => float.IsInfinity(value.m_real) || float.IsInfinity(value.m_imaginary);

        public static bool IsNaN(ComplexF value) => !IsInfinity(value) && !IsFinite(value);

        public static ComplexF Log(ComplexF value) {
            return new ComplexF(MathF.Log(Abs(value)), MathF.Atan2(value.m_imaginary, value.m_real));
        }

        public static ComplexF Log(ComplexF value, float baseValue) {
            return Log(value) / Log(baseValue);
        }

        public static ComplexF Log10(ComplexF value) {
            ComplexF tempLog = Log(value);
            return Scale(tempLog, InverseOfLog10);
        }

        public static ComplexF Exp(ComplexF value) {
            float expReal = MathF.Exp(value.m_real);
            (float sin, float cos) = MathF.SinCos(value.m_imaginary);
            float cosImaginary = expReal * cos;
            float sinImaginary = expReal * sin;
            return new ComplexF(cosImaginary, sinImaginary);
        }

        public static ComplexF Sqrt(ComplexF value) {

            if (value.m_imaginary == 0.0F) {
                // Handle the trivial case quickly.
                if (value.m_real < 0.0) {
                    return new ComplexF(0.0F, MathF.Sqrt(-value.m_real));
                }

                return new ComplexF(MathF.Sqrt(value.m_real), 0.0F);
            }

            // One way to compute Sqrt(z) is just to call Pow(z, 0.5), which coverts to polar coordinates
            // (sqrt + atan), halves the phase, and reconverts to cartesian coordinates (cos + sin).
            // Not only is this more expensive than necessary, it also fails to preserve certain expected
            // symmetries, such as that the square root of a pure negative is a pure imaginary, and that the
            // square root of a pure imaginary has exactly equal real and imaginary parts. This all goes
            // back to the fact that Math.PI is not stored with infinite precision, so taking half of Math.PI
            // does not land us on an argument with cosine exactly equal to zero.

            // To find a fast and symmetry-respecting formula for complex square root,
            // note x + i y = \sqrt{a + i b} implies x^2 + 2 i x y - y^2 = a + i b,
            // so x^2 - y^2 = a and 2 x y = b. Cross-substitute and use the quadratic formula to obtain
            //   x = \sqrt{\frac{\sqrt{a^2 + b^2} + a}{2}}  y = \pm \sqrt{\frac{\sqrt{a^2 + b^2} - a}{2}}
            // There is just one complication: depending on the sign on a, either x or y suffers from
            // cancelation when |b| << |a|. We can get around this by noting that our formulas imply
            // x^2 y^2 = b^2 / 4, so |x| |y| = |b| / 2. So after computing the one that doesn't suffer
            // from cancelation, we can compute the other with just a division. This is basically just
            // the right way to evaluate the quadratic formula without cancelation.

            // All this reduces our total cost to two sqrts and a few flops, and it respects the desired
            // symmetries. Much better than atan + cos + sin!

            // The signs are a matter of choice of branch cut, which is traditionally taken so x > 0 and sign(y) = sign(b).

            // If the components are too large, Hypot will overflow, even though the subsequent sqrt would
            // make the result representable. To avoid this, we re-scale (by exact powers of 2 for accuracy)
            // when we encounter very large components to avoid intermediate infinities.
            bool rescale = false;
            float realCopy = value.m_real;
            float imaginaryCopy = value.m_imaginary;
            if ((MathF.Abs(realCopy) >= s_sqrtRescaleThreshold) || (MathF.Abs(imaginaryCopy) >= s_sqrtRescaleThreshold)) {
                if (float.IsInfinity(value.m_imaginary) && !float.IsNaN(value.m_real)) {
                    // We need to handle infinite imaginary parts specially because otherwise
                    // our formulas below produce inf/inf = NaN. The NaN test is necessary
                    // so that we return NaN rather than (+inf,inf) for (NaN,inf).
                    return (new ComplexF(float.PositiveInfinity, imaginaryCopy));
                }

                realCopy *= 0.25F;
                imaginaryCopy *= 0.25F;
                rescale = true;
            }

            // This is the core of the algorithm. Everything else is special case handling.
            float x, y;
            if (realCopy >= 0.0F) {
                x = MathF.Sqrt((Hypot(realCopy, imaginaryCopy) + realCopy) * 0.5F);
                y = imaginaryCopy / (2.0F * x);
            } else {
                y = MathF.Sqrt((Hypot(realCopy, imaginaryCopy) - realCopy) * 0.5F);
                if (imaginaryCopy < 0.0F) y = -y;
                x = imaginaryCopy / (2.0F * y);
            }

            if (rescale) {
                x *= 2.0F;
                y *= 2.0F;
            }

            return new ComplexF(x, y);
        }

        public static ComplexF Pow(ComplexF value, ComplexF power) {
            if (power == Zero) {
                return One;
            }

            if (value == Zero) {
                return Zero;
            }

            float valueReal = value.m_real;
            float valueImaginary = value.m_imaginary;
            float powerReal = power.m_real;
            float powerImaginary = power.m_imaginary;

            float rho = Abs(value);
            float theta = MathF.Atan2(valueImaginary, valueReal);
            float newRho = powerReal * theta + powerImaginary * MathF.Log(rho);

            float t = MathF.Pow(rho, powerReal) * MathF.Pow(MathF.E, -powerImaginary * theta);

            return new ComplexF(t * MathF.Cos(newRho), t * MathF.Sin(newRho));
        }

        public static ComplexF Pow(ComplexF value, float power) {
            return Pow(value, new ComplexF(power, 0));
        }

        private static ComplexF Scale(ComplexF value, float factor) {
            float realResult = factor * value.m_real;
            float imaginaryResuilt = factor * value.m_imaginary;
            return new ComplexF(realResult, imaginaryResuilt);
        }

        //
        // Explicit Conversions To ComplexF
        //

        public static explicit operator ComplexF(decimal value) {
            return new ComplexF((float)value, 0.0F);
        }

        /// <summary>Explicitly converts a <see cref="Int128" /> value to a double-precision complex number.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to a double-precision complex number.</returns>
        public static explicit operator ComplexF(Int128 value) {
            return new ComplexF((float)value, 0.0F);
        }

        public static explicit operator ComplexF(BigInteger value) {
            return new ComplexF((float)value, 0.0F);
        }

        /// <summary>Explicitly converts a <see cref="UInt128" /> value to a double-precision complex number.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to a double-precision complex number.</returns>
        [CLSCompliant(false)]
        public static explicit operator ComplexF(UInt128 value) {
            return new ComplexF((float)value, 0.0F);
        }

        //
        // Implicit Conversions To ComplexF
        //

        public static implicit operator ComplexF(byte value) {
            return new ComplexF(value, 0.0F);
        }

        /// <summary>Implicitly converts a <see cref="char" /> value to a double-precision complex number.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to a double-precision complex number.</returns>
        public static implicit operator ComplexF(char value) {
            return new ComplexF(value, 0.0F);
        }

        public static explicit operator ComplexF(double value) {
            return new ComplexF((float)value, 0.0F);
        }

        /// <summary>Implicitly converts a <see cref="Half" /> value to a double-precision complex number.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to a double-precision complex number.</returns>
        public static implicit operator ComplexF(Half value) {
            return new ComplexF((float)value, 0.0F);
        }

        public static implicit operator ComplexF(short value) {
            return new ComplexF(value, 0.0F);
        }

        public static implicit operator ComplexF(int value) {
            return new ComplexF(value, 0.0F);
        }

        public static implicit operator ComplexF(long value) {
            return new ComplexF(value, 0.0F);
        }

        /// <summary>Implicitly converts a <see cref="IntPtr" /> value to a double-precision complex number.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to a double-precision complex number.</returns>
        public static implicit operator ComplexF(nint value) {
            return new ComplexF(value, 0.0F);
        }

        [CLSCompliant(false)]
        public static implicit operator ComplexF(sbyte value) {
            return new ComplexF(value, 0.0F);
        }

        public static implicit operator ComplexF(float value) {
            return new ComplexF(value, 0.0F);
        }

        [CLSCompliant(false)]
        public static implicit operator ComplexF(ushort value) {
            return new ComplexF(value, 0.0F);
        }

        [CLSCompliant(false)]
        public static implicit operator ComplexF(uint value) {
            return new ComplexF(value, 0.0F);
        }

        [CLSCompliant(false)]
        public static implicit operator ComplexF(ulong value) {
            return new ComplexF(value, 0.0F);
        }

        /// <summary>Implicitly converts a <see cref="UIntPtr" /> value to a double-precision complex number.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to a double-precision complex number.</returns>
        [CLSCompliant(false)]
        public static implicit operator ComplexF(nuint value) {
            return new ComplexF(value, 0.0F);
        }

        //
        // IAdditiveIdentity
        //

        /// <inheritdoc cref="IAdditiveIdentity{TSelf, TResult}.AdditiveIdentity" />
        static ComplexF IAdditiveIdentity<ComplexF, ComplexF>.AdditiveIdentity => new ComplexF(0.0F, 0.0F);

        //
        // IDecrementOperators
        //

        /// <inheritdoc cref="IDecrementOperators{TSelf}.op_Decrement(TSelf)" />
        public static ComplexF operator --(ComplexF value) => value - One;

        //
        // IIncrementOperators
        //

        /// <inheritdoc cref="IIncrementOperators{TSelf}.op_Increment(TSelf)" />
        public static ComplexF operator ++(ComplexF value) => value + One;

        //
        // IMultiplicativeIdentity
        //

        /// <inheritdoc cref="IMultiplicativeIdentity{TSelf, TResult}.MultiplicativeIdentity" />
        static ComplexF IMultiplicativeIdentity<ComplexF, ComplexF>.MultiplicativeIdentity => new ComplexF(1.0F, 0.0F);

        //
        // INumberBase
        //

        /// <inheritdoc cref="INumberBase{TSelf}.One" />
        static ComplexF INumberBase<ComplexF>.One => new ComplexF(1.0F, 0.0F);

        /// <inheritdoc cref="INumberBase{TSelf}.Radix" />
        static int INumberBase<ComplexF>.Radix => 2;

        /// <inheritdoc cref="INumberBase{TSelf}.Zero" />
        static ComplexF INumberBase<ComplexF>.Zero => new ComplexF(0.0F, 0.0F);

        /// <inheritdoc cref="INumberBase{TSelf}.Abs(TSelf)" />
        static ComplexF INumberBase<ComplexF>.Abs(ComplexF value) => Abs(value);

        /// <inheritdoc cref="INumberBase{TSelf}.CreateChecked{TOther}(TOther)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ComplexF CreateChecked<TOther>(TOther value)
            where TOther : INumberBase<TOther> {
            ComplexF result;

            if (typeof(TOther) == typeof(ComplexF)) {
                result = (ComplexF)(object)value;
            } else if (!TryConvertFrom(value, out result) && !TOther.TryConvertToChecked(value, out result)) {
                ThrowHelper.ThrowNotSupportedException();
            }

            return result;
        }

        /// <inheritdoc cref="INumberBase{TSelf}.CreateSaturating{TOther}(TOther)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ComplexF CreateSaturating<TOther>(TOther value)
            where TOther : INumberBase<TOther> {
            ComplexF result;

            if (typeof(TOther) == typeof(ComplexF)) {
                result = (ComplexF)(object)value;
            } else if (!TryConvertFrom(value, out result) && !TOther.TryConvertToSaturating(value, out result)) {
                ThrowHelper.ThrowNotSupportedException();
            }

            return result;
        }

        /// <inheritdoc cref="INumberBase{TSelf}.CreateTruncating{TOther}(TOther)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ComplexF CreateTruncating<TOther>(TOther value)
            where TOther : INumberBase<TOther> {
            ComplexF result;

            if (typeof(TOther) == typeof(ComplexF)) {
                result = (ComplexF)(object)value;
            } else if (!TryConvertFrom(value, out result) && !TOther.TryConvertToTruncating(value, out result)) {
                ThrowHelper.ThrowNotSupportedException();
            }

            return result;
        }

        /// <inheritdoc cref="INumberBase{TSelf}.IsCanonical(TSelf)" />
        static bool INumberBase<ComplexF>.IsCanonical(ComplexF value) => true;

        /// <inheritdoc cref="INumberBase{TSelf}.IsComplexNumber(TSelf)" />
        public static bool IsComplexNumber(ComplexF value) => (value.m_real != 0.0) && (value.m_imaginary != 0.0);

        /// <inheritdoc cref="INumberBase{TSelf}.IsEvenInteger(TSelf)" />
        public static bool IsEvenInteger(ComplexF value) => (value.m_imaginary == 0) && float.IsEvenInteger(value.m_real);

        /// <inheritdoc cref="INumberBase{TSelf}.IsImaginaryNumber(TSelf)" />
        public static bool IsImaginaryNumber(ComplexF value) => (value.m_real == 0.0) && float.IsRealNumber(value.m_imaginary);

        /// <inheritdoc cref="INumberBase{TSelf}.IsInteger(TSelf)" />
        public static bool IsInteger(ComplexF value) => (value.m_imaginary == 0) && float.IsInteger(value.m_real);

        /// <inheritdoc cref="INumberBase{TSelf}.IsNegative(TSelf)" />
        public static bool IsNegative(ComplexF value) {
            // since complex numbers do not have a well-defined concept of
            // negative we report false if this value has an imaginary part

            return (value.m_imaginary == 0.0) && float.IsNegative(value.m_real);
        }

        /// <inheritdoc cref="INumberBase{TSelf}.IsNegativeInfinity(TSelf)" />
        public static bool IsNegativeInfinity(ComplexF value) {
            // since complex numbers do not have a well-defined concept of
            // negative we report false if this value has an imaginary part

            return (value.m_imaginary == 0.0) && float.IsNegativeInfinity(value.m_real);
        }

        /// <inheritdoc cref="INumberBase{TSelf}.IsNormal(TSelf)" />
        public static bool IsNormal(ComplexF value) {
            // much as IsFinite requires both part to be finite, we require both
            // part to be "normal" (finite, non-zero, and non-subnormal) to be true

            return float.IsNormal(value.m_real)
                && ((value.m_imaginary == 0.0) || float.IsNormal(value.m_imaginary));
        }

        /// <inheritdoc cref="INumberBase{TSelf}.IsOddInteger(TSelf)" />
        public static bool IsOddInteger(ComplexF value) => (value.m_imaginary == 0) && float.IsOddInteger(value.m_real);

        /// <inheritdoc cref="INumberBase{TSelf}.IsPositive(TSelf)" />
        public static bool IsPositive(ComplexF value) {
            // since complex numbers do not have a well-defined concept of
            // negative we report false if this value has an imaginary part

            return (value.m_imaginary == 0.0) && float.IsPositive(value.m_real);
        }

        /// <inheritdoc cref="INumberBase{TSelf}.IsPositiveInfinity(TSelf)" />
        public static bool IsPositiveInfinity(ComplexF value) {
            // since complex numbers do not have a well-defined concept of
            // positive we report false if this value has an imaginary part

            return (value.m_imaginary == 0.0) && float.IsPositiveInfinity(value.m_real);
        }

        /// <inheritdoc cref="INumberBase{TSelf}.IsRealNumber(TSelf)" />
        public static bool IsRealNumber(ComplexF value) => (value.m_imaginary == 0.0) && float.IsRealNumber(value.m_real);

        /// <inheritdoc cref="INumberBase{TSelf}.IsSubnormal(TSelf)" />
        public static bool IsSubnormal(ComplexF value) {
            // much as IsInfinite allows either part to be infinite, we allow either
            // part to be "subnormal" (finite, non-zero, and non-normal) to be true

            return float.IsSubnormal(value.m_real) || float.IsSubnormal(value.m_imaginary);
        }

        /// <inheritdoc cref="INumberBase{TSelf}.IsZero(TSelf)" />
        static bool INumberBase<ComplexF>.IsZero(ComplexF value) => (value.m_real == 0.0) && (value.m_imaginary == 0.0);

        /// <inheritdoc cref="INumberBase{TSelf}.MaxMagnitude(TSelf, TSelf)" />
        public static ComplexF MaxMagnitude(ComplexF x, ComplexF y) {
            // complex numbers are not normally comparable, however every complex
            // number has a real magnitude (absolute value) and so we can provide
            // an implementation for MaxMagnitude

            // This matches the IEEE 754:2019 `maximumMagnitude` function
            //
            // It propagates NaN inputs back to the caller and
            // otherwise returns the input with a larger magnitude.
            // It treats +0 as larger than -0 as per the specification.

            float ax = Abs(x);
            float ay = Abs(y);

            if ((ax > ay) || float.IsNaN(ax)) {
                return x;
            }

            if (ax == ay) {
                // We have two equal magnitudes which means we have two of the following
                //   `+a + ib`
                //   `-a + ib`
                //   `+a - ib`
                //   `-a - ib`
                //
                // We want to treat `+a + ib` as greater than everything and `-a - ib` as
                // lesser. For `-a + ib` and `+a - ib` its "ambiguous" which should be preferred
                // so we will just preference `+a - ib` since that's the most correct choice
                // in the face of something like `+a - i0.0` vs `-a + i0.0`. This is the "most
                // correct" choice because both represent real numbers and `+a` is preferred
                // over `-a`.

                if (float.IsNegative(y.m_real)) {
                    if (float.IsNegative(y.m_imaginary)) {
                        // when `y` is `-a - ib` we always prefer `x` (its either the same as
                        // `x` or some part of `x` is positive).

                        return x;
                    } else {
                        if (float.IsNegative(x.m_real)) {
                            // when `y` is `-a + ib` and `x` is `-a + ib` or `-a - ib` then
                            // we either have same value or both parts of `x` are negative
                            // and we want to prefer `y`.

                            return y;
                        } else {
                            // when `y` is `-a + ib` and `x` is `+a + ib` or `+a - ib` then
                            // we want to prefer `x` because either both parts are positive
                            // or we want to prefer `+a - ib` due to how it handles when `x`
                            // represents a real number.

                            return x;
                        }
                    }
                } else if (float.IsNegative(y.m_imaginary)) {
                    if (float.IsNegative(x.m_real)) {
                        // when `y` is `+a - ib` and `x` is `-a + ib` or `-a - ib` then
                        // we either both parts of `x` are negative or we want to prefer
                        // `+a - ib` due to how it handles when `y` represents a real number.

                        return y;
                    } else {
                        // when `y` is `+a - ib` and `x` is `+a + ib` or `+a - ib` then
                        // we want to prefer `x` because either both parts are positive
                        // or they represent the same value.

                        return x;
                    }
                }
            }

            return y;
        }

        /// <inheritdoc cref="INumberBase{TSelf}.MaxMagnitudeNumber(TSelf, TSelf)" />
        static ComplexF INumberBase<ComplexF>.MaxMagnitudeNumber(ComplexF x, ComplexF y) {
            // complex numbers are not normally comparable, however every complex
            // number has a real magnitude (absolute value) and so we can provide
            // an implementation for MaxMagnitudeNumber

            // This matches the IEEE 754:2019 `maximumMagnitudeNumber` function
            //
            // It does not propagate NaN inputs back to the caller and
            // otherwise returns the input with a larger magnitude.
            // It treats +0 as larger than -0 as per the specification.

            float ax = Abs(x);
            float ay = Abs(y);

            if ((ax > ay) || float.IsNaN(ay)) {
                return x;
            }

            if (ax == ay) {
                // We have two equal magnitudes which means we have two of the following
                //   `+a + ib`
                //   `-a + ib`
                //   `+a - ib`
                //   `-a - ib`
                //
                // We want to treat `+a + ib` as greater than everything and `-a - ib` as
                // lesser. For `-a + ib` and `+a - ib` its "ambiguous" which should be preferred
                // so we will just preference `+a - ib` since that's the most correct choice
                // in the face of something like `+a - i0.0` vs `-a + i0.0`. This is the "most
                // correct" choice because both represent real numbers and `+a` is preferred
                // over `-a`.

                if (float.IsNegative(y.m_real)) {
                    if (float.IsNegative(y.m_imaginary)) {
                        // when `y` is `-a - ib` we always prefer `x` (its either the same as
                        // `x` or some part of `x` is positive).

                        return x;
                    } else {
                        if (float.IsNegative(x.m_real)) {
                            // when `y` is `-a + ib` and `x` is `-a + ib` or `-a - ib` then
                            // we either have same value or both parts of `x` are negative
                            // and we want to prefer `y`.

                            return y;
                        } else {
                            // when `y` is `-a + ib` and `x` is `+a + ib` or `+a - ib` then
                            // we want to prefer `x` because either both parts are positive
                            // or we want to prefer `+a - ib` due to how it handles when `x`
                            // represents a real number.

                            return x;
                        }
                    }
                } else if (float.IsNegative(y.m_imaginary)) {
                    if (float.IsNegative(x.m_real)) {
                        // when `y` is `+a - ib` and `x` is `-a + ib` or `-a - ib` then
                        // we either both parts of `x` are negative or we want to prefer
                        // `+a - ib` due to how it handles when `y` represents a real number.

                        return y;
                    } else {
                        // when `y` is `+a - ib` and `x` is `+a + ib` or `+a - ib` then
                        // we want to prefer `x` because either both parts are positive
                        // or they represent the same value.

                        return x;
                    }
                }
            }

            return y;
        }

        /// <inheritdoc cref="INumberBase{TSelf}.MinMagnitude(TSelf, TSelf)" />
        public static ComplexF MinMagnitude(ComplexF x, ComplexF y) {
            // complex numbers are not normally comparable, however every complex
            // number has a real magnitude (absolute value) and so we can provide
            // an implementation for MaxMagnitude

            // This matches the IEEE 754:2019 `minimumMagnitude` function
            //
            // It propagates NaN inputs back to the caller and
            // otherwise returns the input with a smaller magnitude.
            // It treats -0 as smaller than +0 as per the specification.

            float ax = Abs(x);
            float ay = Abs(y);

            if ((ax < ay) || float.IsNaN(ax)) {
                return x;
            }

            if (ax == ay) {
                // We have two equal magnitudes which means we have two of the following
                //   `+a + ib`
                //   `-a + ib`
                //   `+a - ib`
                //   `-a - ib`
                //
                // We want to treat `+a + ib` as greater than everything and `-a - ib` as
                // lesser. For `-a + ib` and `+a - ib` its "ambiguous" which should be preferred
                // so we will just preference `-a + ib` since that's the most correct choice
                // in the face of something like `+a - i0.0` vs `-a + i0.0`. This is the "most
                // correct" choice because both represent real numbers and `-a` is preferred
                // over `+a`.

                if (float.IsNegative(y.m_real)) {
                    if (float.IsNegative(y.m_imaginary)) {
                        // when `y` is `-a - ib` we always prefer `y` as both parts are negative
                        return y;
                    } else {
                        if (float.IsNegative(x.m_real)) {
                            // when `y` is `-a + ib` and `x` is `-a + ib` or `-a - ib` then
                            // we either have same value or both parts of `x` are negative
                            // and we want to prefer it.

                            return x;
                        } else {
                            // when `y` is `-a + ib` and `x` is `+a + ib` or `+a - ib` then
                            // we want to prefer `y` because either both parts of 'x' are positive
                            // or we want to prefer `-a - ib` due to how it handles when `y`
                            // represents a real number.

                            return y;
                        }
                    }
                } else if (float.IsNegative(y.m_imaginary)) {
                    if (float.IsNegative(x.m_real)) {
                        // when `y` is `+a - ib` and `x` is `-a + ib` or `-a - ib` then
                        // either both parts of `x` are negative or we want to prefer
                        // `-a - ib` due to how it handles when `x` represents a real number.

                        return x;
                    } else {
                        // when `y` is `+a - ib` and `x` is `+a + ib` or `+a - ib` then
                        // we want to prefer `y` because either both parts of x are positive
                        // or they represent the same value.

                        return y;
                    }
                } else {
                    return x;
                }
            }

            return y;
        }

        /// <inheritdoc cref="INumberBase{TSelf}.MinMagnitudeNumber(TSelf, TSelf)" />
        static ComplexF INumberBase<ComplexF>.MinMagnitudeNumber(ComplexF x, ComplexF y) {
            // complex numbers are not normally comparable, however every complex
            // number has a real magnitude (absolute value) and so we can provide
            // an implementation for MinMagnitudeNumber

            // This matches the IEEE 754:2019 `minimumMagnitudeNumber` function
            //
            // It does not propagate NaN inputs back to the caller and
            // otherwise returns the input with a smaller magnitude.
            // It treats -0 as smaller than +0 as per the specification.

            float ax = Abs(x);
            float ay = Abs(y);

            if ((ax < ay) || float.IsNaN(ay)) {
                return x;
            }

            if (ax == ay) {
                // We have two equal magnitudes which means we have two of the following
                //   `+a + ib`
                //   `-a + ib`
                //   `+a - ib`
                //   `-a - ib`
                //
                // We want to treat `+a + ib` as greater than everything and `-a - ib` as
                // lesser. For `-a + ib` and `+a - ib` its "ambiguous" which should be preferred
                // so we will just preference `-a + ib` since that's the most correct choice
                // in the face of something like `+a - i0.0` vs `-a + i0.0`. This is the "most
                // correct" choice because both represent real numbers and `-a` is preferred
                // over `+a`.

                if (float.IsNegative(y.m_real)) {
                    if (float.IsNegative(y.m_imaginary)) {
                        // when `y` is `-a - ib` we always prefer `y` as both parts are negative
                        return y;
                    } else {
                        if (float.IsNegative(x.m_real)) {
                            // when `y` is `-a + ib` and `x` is `-a + ib` or `-a - ib` then
                            // we either have same value or both parts of `x` are negative
                            // and we want to prefer it.

                            return x;
                        } else {
                            // when `y` is `-a + ib` and `x` is `+a + ib` or `+a - ib` then
                            // we want to prefer `y` because either both parts of 'x' are positive
                            // or we want to prefer `-a - ib` due to how it handles when `y`
                            // represents a real number.

                            return y;
                        }
                    }
                } else if (float.IsNegative(y.m_imaginary)) {
                    if (float.IsNegative(x.m_real)) {
                        // when `y` is `+a - ib` and `x` is `-a + ib` or `-a - ib` then
                        // either both parts of `x` are negative or we want to prefer
                        // `-a - ib` due to how it handles when `x` represents a real number.

                        return x;
                    } else {
                        // when `y` is `+a - ib` and `x` is `+a + ib` or `+a - ib` then
                        // we want to prefer `y` because either both parts of x are positive
                        // or they represent the same value.

                        return y;
                    }
                } else {
                    return x;
                }
            }

            return y;
        }

        /// <inheritdoc cref="INumberBase{TSelf}.Parse(ReadOnlySpan{char}, NumberStyles, IFormatProvider?)" />
        public static ComplexF Parse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider) {
            if (!TryParse(s, style, provider, out ComplexF result)) {
                ThrowHelper.ThrowOverflowException();
            }
            return result;
        }

        /// <inheritdoc cref="INumberBase{TSelf}.Parse(string, NumberStyles, IFormatProvider?)" />
        public static ComplexF Parse(string s, NumberStyles style, IFormatProvider? provider) {
            ArgumentNullException.ThrowIfNull(s);
            return Parse(s.AsSpan(), style, provider);
        }

        /// <inheritdoc cref="INumberBase{TSelf}.TryConvertFromChecked{TOther}(TOther, out TSelf)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool INumberBase<ComplexF>.TryConvertFromChecked<TOther>(TOther value, out ComplexF result) {
            return TryConvertFrom<TOther>(value, out result);
        }

        /// <inheritdoc cref="INumberBase{TSelf}.TryConvertFromSaturating{TOther}(TOther, out TSelf)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool INumberBase<ComplexF>.TryConvertFromSaturating<TOther>(TOther value, out ComplexF result) {
            return TryConvertFrom<TOther>(value, out result);
        }

        /// <inheritdoc cref="INumberBase{TSelf}.TryConvertFromTruncating{TOther}(TOther, out TSelf)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool INumberBase<ComplexF>.TryConvertFromTruncating<TOther>(TOther value, out ComplexF result) {
            return TryConvertFrom<TOther>(value, out result);
        }

        private static bool TryConvertFrom<TOther>(TOther value, out ComplexF result)
            where TOther : INumberBase<TOther> {
            // We don't want to defer to `float.Create*(value)` because some type might have its own
            // `TOther.ConvertTo*(value, out ComplexF result)` handling that would end up bypassed.

            if (typeof(TOther) == typeof(byte)) {
                byte actualValue = (byte)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(char)) {
                char actualValue = (char)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(decimal)) {
                decimal actualValue = (decimal)(object)value;
                result = (ComplexF)actualValue;
                return true;
            } else if (typeof(TOther) == typeof(double)) {
                double actualValue = (double)(object)value;
                result = (ComplexF)actualValue;
                return true;
            } else if (typeof(TOther) == typeof(Half)) {
                Half actualValue = (Half)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(short)) {
                short actualValue = (short)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(int)) {
                int actualValue = (int)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(long)) {
                long actualValue = (long)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(Int128)) {
                Int128 actualValue = (Int128)(object)value;
                result = (ComplexF)actualValue;
                return true;
            } else if (typeof(TOther) == typeof(nint)) {
                nint actualValue = (nint)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(sbyte)) {
                sbyte actualValue = (sbyte)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(float)) {
                float actualValue = (float)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(ushort)) {
                ushort actualValue = (ushort)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(uint)) {
                uint actualValue = (uint)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(ulong)) {
                ulong actualValue = (ulong)(object)value;
                result = actualValue;
                return true;
            } else if (typeof(TOther) == typeof(UInt128)) {
                UInt128 actualValue = (UInt128)(object)value;
                result = (ComplexF)actualValue;
                return true;
            } else if (typeof(TOther) == typeof(nuint)) {
                nuint actualValue = (nuint)(object)value;
                result = actualValue;
                return true;
            } else {
                result = default;
                return false;
            }
        }

        /// <inheritdoc cref="INumberBase{TSelf}.TryConvertToChecked{TOther}(TSelf, out TOther)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool INumberBase<ComplexF>.TryConvertToChecked<TOther>(ComplexF value, [MaybeNullWhen(false)] out TOther result) {
            // ComplexF numbers with an imaginary part can't be represented as a "real number"
            // so we'll throw an OverflowException for this scenario for integer types and
            // for decimal. However, we will convert it to NaN for the floating-point types,
            // since that's what Sqrt(-1) (which is `new ComplexF(0, 1)`) results in.

            if (typeof(TOther) == typeof(byte)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                byte actualResult = checked((byte)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(char)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                char actualResult = checked((char)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(decimal)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                decimal actualResult = checked((decimal)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(double)) {
                double actualResult = (value.m_imaginary != 0) ? float.NaN : value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(Half)) {
                Half actualResult = (value.m_imaginary != 0) ? Half.NaN : (Half)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(short)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                short actualResult = checked((short)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(int)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                int actualResult = checked((int)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(long)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                long actualResult = checked((long)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(Int128)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                Int128 actualResult = checked((Int128)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(nint)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                nint actualResult = checked((nint)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(BigInteger)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                BigInteger actualResult = checked((BigInteger)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(sbyte)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                sbyte actualResult = checked((sbyte)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(float)) {
                float actualResult = (value.m_imaginary != 0) ? float.NaN : (float)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(ushort)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                ushort actualResult = checked((ushort)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(uint)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                uint actualResult = checked((uint)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(ulong)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                ulong actualResult = checked((ulong)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(UInt128)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                UInt128 actualResult = checked((UInt128)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(nuint)) {
                if (value.m_imaginary != 0) {
                    ThrowHelper.ThrowOverflowException();
                }

                nuint actualResult = checked((nuint)value.m_real);
                result = (TOther)(object)actualResult;
                return true;
            } else {
                result = default;
                return false;
            }
        }

        /// <inheritdoc cref="INumberBase{TSelf}.TryConvertToSaturating{TOther}(TSelf, out TOther)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool INumberBase<ComplexF>.TryConvertToSaturating<TOther>(ComplexF value, [MaybeNullWhen(false)] out TOther result) {
            // ComplexF numbers with an imaginary part can't be represented as a "real number"
            // and there isn't really a well-defined way to "saturate" to just a real value.
            //
            // The two potential options are that we either treat complex numbers with a non-
            // zero imaginary part as NaN and then convert that to 0 -or- we ignore the imaginary
            // part and only consider the real part.
            //
            // We use the latter below since that is "more useful" given an unknown number type.
            // Users who want 0 instead can always check `IsComplexNumber` and special-case the
            // handling.

            if (typeof(TOther) == typeof(byte)) {
                byte actualResult = (value.m_real >= byte.MaxValue) ? byte.MaxValue :
                                    (value.m_real <= byte.MinValue) ? byte.MinValue : (byte)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(char)) {
                char actualResult = (value.m_real >= char.MaxValue) ? char.MaxValue :
                                    (value.m_real <= char.MinValue) ? char.MinValue : (char)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(decimal)) {
                decimal actualResult = (value.m_real >= (double)decimal.MaxValue) ? decimal.MaxValue :
                                       (value.m_real <= (double)decimal.MinValue) ? decimal.MinValue : (decimal)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(double)) {
                double actualResult = value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(Half)) {
                Half actualResult = (Half)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(short)) {
                short actualResult = (value.m_real >= short.MaxValue) ? short.MaxValue :
                                     (value.m_real <= short.MinValue) ? short.MinValue : (short)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(int)) {
                int actualResult = (value.m_real >= int.MaxValue) ? int.MaxValue :
                                   (value.m_real <= int.MinValue) ? int.MinValue : (int)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(long)) {
                long actualResult = (value.m_real >= long.MaxValue) ? long.MaxValue :
                                    (value.m_real <= long.MinValue) ? long.MinValue : (long)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(Int128)) {
                Int128 actualResult = (value.m_real >= +170141183460469231731687303715884105727.0) ? Int128.MaxValue :
                                      (value.m_real <= -170141183460469231731687303715884105728.0) ? Int128.MinValue : (Int128)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(nint)) {
                nint actualResult = (value.m_real >= nint.MaxValue) ? nint.MaxValue :
                                    (value.m_real <= nint.MinValue) ? nint.MinValue : (nint)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(BigInteger)) {
                BigInteger actualResult = (BigInteger)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(sbyte)) {
                sbyte actualResult = (value.m_real >= sbyte.MaxValue) ? sbyte.MaxValue :
                                     (value.m_real <= sbyte.MinValue) ? sbyte.MinValue : (sbyte)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(float)) {
                float actualResult = (float)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(ushort)) {
                ushort actualResult = (value.m_real >= ushort.MaxValue) ? ushort.MaxValue :
                                      (value.m_real <= ushort.MinValue) ? ushort.MinValue : (ushort)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(uint)) {
                uint actualResult = (value.m_real >= uint.MaxValue) ? uint.MaxValue :
                                    (value.m_real <= uint.MinValue) ? uint.MinValue : (uint)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(ulong)) {
                ulong actualResult = (value.m_real >= ulong.MaxValue) ? ulong.MaxValue :
                                     (value.m_real <= ulong.MinValue) ? ulong.MinValue : (ulong)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(UInt128)) {
                UInt128 actualResult = (value.m_real >= 340282366920938463463374607431768211455.0) ? UInt128.MaxValue :
                                       (value.m_real <= 0.0) ? UInt128.MinValue : (UInt128)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(nuint)) {
                nuint actualResult = (value.m_real >= nuint.MaxValue) ? nuint.MaxValue :
                                     (value.m_real <= nuint.MinValue) ? nuint.MinValue : (nuint)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else {
                result = default;
                return false;
            }
        }

        /// <inheritdoc cref="INumberBase{TSelf}.TryConvertToTruncating{TOther}(TSelf, out TOther)" />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool INumberBase<ComplexF>.TryConvertToTruncating<TOther>(ComplexF value, [MaybeNullWhen(false)] out TOther result) {
            // ComplexF numbers with an imaginary part can't be represented as a "real number"
            // so we'll only consider the real part for the purposes of truncation.

            if (typeof(TOther) == typeof(byte)) {
                byte actualResult = (value.m_real >= byte.MaxValue) ? byte.MaxValue :
                                    (value.m_real <= byte.MinValue) ? byte.MinValue : (byte)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(char)) {
                char actualResult = (value.m_real >= char.MaxValue) ? char.MaxValue :
                                    (value.m_real <= char.MinValue) ? char.MinValue : (char)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(decimal)) {
                decimal actualResult = (value.m_real >= (double)decimal.MaxValue) ? decimal.MaxValue :
                                       (value.m_real <= (double)decimal.MinValue) ? decimal.MinValue : (decimal)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(double)) {
                double actualResult = value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(Half)) {
                Half actualResult = (Half)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(short)) {
                short actualResult = (value.m_real >= short.MaxValue) ? short.MaxValue :
                                     (value.m_real <= short.MinValue) ? short.MinValue : (short)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(int)) {
                int actualResult = (value.m_real >= int.MaxValue) ? int.MaxValue :
                                   (value.m_real <= int.MinValue) ? int.MinValue : (int)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(long)) {
                long actualResult = (value.m_real >= long.MaxValue) ? long.MaxValue :
                                    (value.m_real <= long.MinValue) ? long.MinValue : (long)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(Int128)) {
                Int128 actualResult = (value.m_real >= +170141183460469231731687303715884105727.0) ? Int128.MaxValue :
                                      (value.m_real <= -170141183460469231731687303715884105728.0) ? Int128.MinValue : (Int128)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(nint)) {
                nint actualResult = (value.m_real >= nint.MaxValue) ? nint.MaxValue :
                                    (value.m_real <= nint.MinValue) ? nint.MinValue : (nint)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(BigInteger)) {
                BigInteger actualResult = (BigInteger)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(sbyte)) {
                sbyte actualResult = (value.m_real >= sbyte.MaxValue) ? sbyte.MaxValue :
                                     (value.m_real <= sbyte.MinValue) ? sbyte.MinValue : (sbyte)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(float)) {
                float actualResult = (float)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(ushort)) {
                ushort actualResult = (value.m_real >= ushort.MaxValue) ? ushort.MaxValue :
                                      (value.m_real <= ushort.MinValue) ? ushort.MinValue : (ushort)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(uint)) {
                uint actualResult = (value.m_real >= uint.MaxValue) ? uint.MaxValue :
                                    (value.m_real <= uint.MinValue) ? uint.MinValue : (uint)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(ulong)) {
                ulong actualResult = (value.m_real >= ulong.MaxValue) ? ulong.MaxValue :
                                     (value.m_real <= ulong.MinValue) ? ulong.MinValue : (ulong)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(UInt128)) {
                UInt128 actualResult = (value.m_real >= 340282366920938463463374607431768211455.0) ? UInt128.MaxValue :
                                       (value.m_real <= 0.0) ? UInt128.MinValue : (UInt128)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else if (typeof(TOther) == typeof(nuint)) {
                nuint actualResult = (value.m_real >= nuint.MaxValue) ? nuint.MaxValue :
                                     (value.m_real <= nuint.MinValue) ? nuint.MinValue : (nuint)value.m_real;
                result = (TOther)(object)actualResult;
                return true;
            } else {
                result = default;
                return false;
            }
        }

        /// <inheritdoc cref="INumberBase{TSelf}.TryParse(ReadOnlySpan{char}, NumberStyles, IFormatProvider?, out TSelf)" />
        public static bool TryParse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider, out ComplexF result) {
            ValidateParseStyleFloatingPoint(style);

            int openBracket = s.IndexOf('<');
            int semicolon = s.IndexOf(';');
            int closeBracket = s.IndexOf('>');

            if ((s.Length < 5) || (openBracket == -1) || (semicolon == -1) || (closeBracket == -1) || (openBracket > semicolon) || (openBracket > closeBracket) || (semicolon > closeBracket)) {
                // We need at least 5 characters for `<0;0>`
                // We also expect a to find an open bracket, a semicolon, and a closing bracket in that order

                result = default;
                return false;
            }

            if ((openBracket != 0) && (((style & NumberStyles.AllowLeadingWhite) == 0) || !s.Slice(0, openBracket).IsWhiteSpace())) {
                // The opening bracket wasn't the first and we either didn't allow leading whitespace
                // or one of the leading characters wasn't whitespace at all.

                result = default;
                return false;
            }

            if (!float.TryParse(s.Slice(openBracket + 1, semicolon), style, provider, out float real)) {
                result = default;
                return false;
            }

            if (char.IsWhiteSpace(s[semicolon + 1])) {
                // We allow a single whitespace after the semicolon regardless of style, this is so that
                // the output of `ToString` can be correctly parsed by default and values will roundtrip.
                semicolon += 1;
            }

            if (!float.TryParse(s.Slice(semicolon + 1, closeBracket - semicolon), style, provider, out float imaginary)) {
                result = default;
                return false;
            }

            if ((closeBracket != (s.Length - 1)) && (((style & NumberStyles.AllowTrailingWhite) == 0) || !s.Slice(closeBracket).IsWhiteSpace())) {
                // The closing bracket wasn't the last and we either didn't allow trailing whitespace
                // or one of the trailing characters wasn't whitespace at all.

                result = default;
                return false;
            }

            result = new ComplexF(real, imaginary);
            return true;

            static void ValidateParseStyleFloatingPoint(NumberStyles style) {
                // Check for undefined flags or hex number
                if ((style & (InvalidNumberStyles | NumberStyles.AllowHexSpecifier)) != 0) {
                    ThrowInvalid(style);

                    static void ThrowInvalid(NumberStyles value) {
                        if ((value & InvalidNumberStyles) != 0) {
                            throw new ArgumentException(SR.Argument_InvalidNumberStyles, nameof(style));
                        }

                        throw new ArgumentException(SR.Arg_HexStyleNotSupported);
                    }
                }
            }
        }

        /// <inheritdoc cref="INumberBase{TSelf}.TryParse(string, NumberStyles, IFormatProvider?, out TSelf)" />
        public static bool TryParse([NotNullWhen(true)] string? s, NumberStyles style, IFormatProvider? provider, out ComplexF result) {
            if (s is null) {
                result = default;
                return false;
            }
            return TryParse(s.AsSpan(), style, provider, out result);
        }

        //
        // IParsable
        //

        /// <inheritdoc cref="IParsable{TSelf}.Parse(string, IFormatProvider?)" />
        public static ComplexF Parse(string s, IFormatProvider? provider) => Parse(s, DefaultNumberStyle, provider);

        /// <inheritdoc cref="IParsable{TSelf}.TryParse(string?, IFormatProvider?, out TSelf)" />
        public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, out ComplexF result) => TryParse(s, DefaultNumberStyle, provider, out result);

        //
        // ISignedNumber
        //

        /// <inheritdoc cref="ISignedNumber{TSelf}.NegativeOne" />
        static ComplexF ISignedNumber<ComplexF>.NegativeOne => new ComplexF(-1.0F, 0.0F);

        //
        // ISpanFormattable
        //

        /// <inheritdoc cref="ISpanFormattable.TryFormat(Span{char}, out int, ReadOnlySpan{char}, IFormatProvider?)" />
        public bool TryFormat(Span<char> destination, out int charsWritten, [StringSyntax(StringSyntaxAttribute.NumericFormat)] ReadOnlySpan<char> format = default, IFormatProvider? provider = null) =>
            TryFormatCore(destination, out charsWritten, format, provider);

        public bool TryFormat(Span<byte> utf8Destination, out int bytesWritten, [StringSyntax(StringSyntaxAttribute.NumericFormat)] ReadOnlySpan<char> format = default, IFormatProvider? provider = null) =>
            TryFormatCore(utf8Destination, out bytesWritten, format, provider);

        private bool TryFormatCore<TChar>(Span<TChar> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? provider) where TChar : unmanaged, IBinaryInteger<TChar> {
            Debug.Assert(typeof(TChar) == typeof(char) || typeof(TChar) == typeof(byte));

            // We have at least 6 more characters for: <0; 0>
            if (destination.Length >= 6) {
                int realChars;
                if (typeof(TChar) == typeof(char) ?
                    m_real.TryFormat(MemoryMarshal.Cast<TChar, char>(destination.Slice(1)), out realChars, format, provider) :
                    m_real.TryFormat(MemoryMarshal.Cast<TChar, byte>(destination.Slice(1)), out realChars, format, provider)) {
                    destination[0] = TChar.CreateTruncating('<');
                    destination = destination.Slice(1 + realChars); // + 1 for <

                    // We have at least 4 more characters for: ; 0>
                    if (destination.Length >= 4) {
                        int imaginaryChars;
                        if (typeof(TChar) == typeof(char) ?
                            m_imaginary.TryFormat(MemoryMarshal.Cast<TChar, char>(destination.Slice(2)), out imaginaryChars, format, provider) :
                            m_imaginary.TryFormat(MemoryMarshal.Cast<TChar, byte>(destination.Slice(2)), out imaginaryChars, format, provider)) {
                            // We have 1 more character for: >
                            if ((uint)(2 + imaginaryChars) < (uint)destination.Length) {
                                destination[0] = TChar.CreateTruncating(';');
                                destination[1] = TChar.CreateTruncating(' ');
                                destination[2 + imaginaryChars] = TChar.CreateTruncating('>');

                                charsWritten = realChars + imaginaryChars + 4;
                                return true;
                            }
                        }
                    }
                }
            }

            charsWritten = 0;
            return false;
        }

        //
        // ISpanParsable
        //

        /// <inheritdoc cref="ISpanParsable{TSelf}.Parse(ReadOnlySpan{char}, IFormatProvider?)" />
        public static ComplexF Parse(ReadOnlySpan<char> s, IFormatProvider? provider) => Parse(s, DefaultNumberStyle, provider);

        /// <inheritdoc cref="ISpanParsable{TSelf}.TryParse(ReadOnlySpan{char}, IFormatProvider?, out TSelf)" />
        public static bool TryParse(ReadOnlySpan<char> s, IFormatProvider? provider, out ComplexF result) => TryParse(s, DefaultNumberStyle, provider, out result);

        //
        // IUnaryPlusOperators
        //

        /// <inheritdoc cref="IUnaryPlusOperators{TSelf, TResult}.op_UnaryPlus(TSelf)" />
        public static ComplexF operator +(ComplexF value) => value;
    }
}
