; ModuleID = '/home/chris/FLL/../FLL-workplace/llvmbugs/info/16069/fail.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@d = common global i32 0, align 4
@c = common global i32* null, align 8
@a = common global i32* null, align 8
@b = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define i32 @foo(i16 signext %p, i32 %q) #0 {
  %1 = alloca i16, align 2
  %2 = alloca i32, align 4
  store i16 %p, i16* %1, align 2
  store i32 %q, i32* %2, align 4
  %3 = load i32* %2, align 4
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %12, label %5

; <label>:5                                       ; preds = %0
  %6 = load i16* %1, align 2
  %7 = sext i16 %6 to i32
  %8 = icmp ne i32 %7, 0
  br i1 %8, label %9, label %15

; <label>:9                                       ; preds = %5
  %10 = load i32* %2, align 4
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %15

; <label>:12                                      ; preds = %9, %0
  %13 = load i16* %1, align 2
  %14 = sext i16 %13 to i32
  br label %18

; <label>:15                                      ; preds = %9, %5
  %16 = load i32* %2, align 4
  %17 = srem i32 1, %16
  br label %18

; <label>:18                                      ; preds = %15, %12
  %19 = phi i32 [ %14, %12 ], [ %17, %15 ]
  ret i32 %19
}

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1
  store i32* @d, i32** @c, align 8
  %2 = load i32** @c, align 8
  store i32 zext (i1 icmp eq (i32 ptrtoint (i32** @a to i32), i32 0) to i32), i32* %2, align 4
  %3 = load i32* @d, align 4
  %4 = call i32 @foo(i16 signext 0, i32 %3)
  store i32 %4, i32* @b, align 4
  ret i32 0
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
