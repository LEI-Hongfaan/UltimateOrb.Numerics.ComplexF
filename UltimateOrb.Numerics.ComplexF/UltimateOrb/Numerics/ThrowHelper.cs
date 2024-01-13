
using System.Diagnostics.CodeAnalysis;

namespace UltimateOrb.Numerics {

    static partial class ThrowHelper {

        [DoesNotReturn]
        internal static void ThrowNotSupportedException() {
            throw new NotSupportedException();
        }

        [DoesNotReturn]
#pragma warning disable CS8763 // A method marked [DoesNotReturn] should not return.
        internal static void ThrowOverflowException() {
            var v = 1u;
            _ = checked(0u - v);
        }
#pragma warning restore CS8763 // A method marked [DoesNotReturn] should not return.
    }
}
