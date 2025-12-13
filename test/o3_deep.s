	.file	"/home/chris/FLL/../FLL-workplace/llvmbugs/info/16069/fail.c"
	.text
	.globl	foo
	.type	foo,@function
foo:                                    # @foo
	.cfi_startproc
# BB#0:
	testl	%esi, %esi
	je	.LBB0_3
# BB#1:
	testw	%di, %di
	je	.LBB0_4
# BB#2:
	cmpl	$1, %esi
	jne	.LBB0_4
.LBB0_3:
	movswl	%di, %edx
	jmp	.LBB0_5
.LBB0_4:
	movl	$1, %eax
	xorl	%edx, %edx
	idivl	%esi
.LBB0_5:
	movl	%edx, %eax
	ret
.Ltmp0:
	.size	foo, .Ltmp0-foo
	.cfi_endproc

	.globl	main
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# BB#0:                                 # %foo.exit
	movl	$a, %ecx
	testl	%ecx, %ecx
	sete	%al
	movzbl	%al, %esi
	xorl	%edi, %edi
	movl	$1, %eax
	xorl	%edx, %edx
	divl	%esi
	testl	%ecx, %ecx
	movq	$d, c(%rip)
	movl	%esi, d(%rip)
	cmovnel	%edi, %edx
	movl	%edx, b(%rip)
	xorl	%eax, %eax
	ret
.Ltmp1:
	.size	main, .Ltmp1-main
	.cfi_endproc

	.type	d,@object               # @d
	.comm	d,4,4
	.type	c,@object               # @c
	.comm	c,8,8
	.type	a,@object               # @a
	.comm	a,8,8
	.type	b,@object               # @b
	.comm	b,4,4

	.section	".note.GNU-stack","",@progbits
