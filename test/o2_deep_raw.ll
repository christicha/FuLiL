; ModuleID = '/home/chris/FLL/../FLL-workplace/llvmbugs/info/15920/fail.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c = common global i8 0, align 1
@i = common global i32 0, align 4
@d = common global i8 0, align 1
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
  %1 = alloca i32, align 4
  %j = alloca i32*, align 8
  store i32 0, i32* %1
  store i8 0, i8* @c, align 1
  br label %2

; <label>:2                                       ; preds = %19, %0
  %3 = load i8* @c, align 1
  %4 = sext i8 %3 to i32
  %5 = icmp sgt i32 %4, -16
  br i1 %5, label %6, label %22

; <label>:6                                       ; preds = %2
  store i32* @i, i32** %j, align 8
  %7 = load i32** %j, align 8
  store i32 1, i32* %7, align 4
  store i8 0, i8* @d, align 1
  br label %8

; <label>:8                                       ; preds = %13, %6
  %9 = load i8* @d, align 1
  %10 = icmp ne i8 %9, 0
  br i1 %10, label %11, label %18

; <label>:11                                      ; preds = %8
  %12 = load i32** %j, align 8
  store i32 0, i32* %12, align 4
  br label %13

; <label>:13                                      ; preds = %11
  %14 = load i8* @d, align 1
  %15 = sext i8 %14 to i32
  %16 = add nsw i32 %15, 0
  %17 = trunc i32 %16 to i8
  store i8 %17, i8* @d, align 1
  br label %8

; <label>:18                                      ; preds = %8
  br label %19

; <label>:19                                      ; preds = %18
  %20 = load i8* @c, align 1
  %21 = add i8 %20, -1
  store i8 %21, i8* @c, align 1
  br label %2

; <label>:22                                      ; preds = %2
  %23 = load i32* @i, align 4
  %24 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32 %23)
  ret i32 0
}

declare i32 @printf(i8*, ...) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
