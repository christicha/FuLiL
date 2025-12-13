; ModuleID = '/home/chris/FLL/../FLL-workplace/llvmbugs/info/16069/fail.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@d = common global i32 0, align 4
@c = common global i32* null, align 8
@a = common global i32* null, align 8
@b = common global i32 0, align 4

; Function Attrs: nounwind readnone uwtable
define i32 @foo(i16 signext %p, i32 %q) #0 {
  %1 = icmp eq i32 %q, 0
  br i1 %1, label %5, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp ne i16 %p, 0
  %4 = icmp eq i32 %q, 1
  %or.cond = and i1 %3, %4
  br i1 %or.cond, label %5, label %7

; <label>:5                                       ; preds = %2, %0
  %6 = sext i16 %p to i32
  br label %9

; <label>:7                                       ; preds = %2
  %8 = srem i32 1, %q
  br label %9

; <label>:9                                       ; preds = %7, %5
  %10 = phi i32 [ %6, %5 ], [ %8, %7 ]
  ret i32 %10
}

; Function Attrs: nounwind uwtable
define i32 @main() #1 {
  store i32* @d, i32** @c, align 8, !tbaa !0
  store i32 zext (i1 icmp eq (i32 ptrtoint (i32** @a to i32), i32 0) to i32), i32* @d, align 4, !tbaa !3
  %1 = tail call i32 @foo(i16 signext 0, i32 zext (i1 icmp eq (i32 ptrtoint (i32** @a to i32), i32 0) to i32))
  store i32 %1, i32* @b, align 4, !tbaa !3
  ret i32 0
}

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = metadata !{metadata !"any pointer", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"int", metadata !1}
