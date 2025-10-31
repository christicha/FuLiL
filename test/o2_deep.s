	.file	"/home/chris/FLL/../FLL-workplace/llvmbugs/info/16069/fail.c"
	.text
	.globl	foo
	.align	16, 0x90
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
	movl	%edx, %eax
	ret
.LBB0_4:
	movl	$1, %eax
	xorl	%edx, %edx
	idivl	%esi
	movl	%edx, %eax
	ret
.Ltmp0:
	.size	foo, .Ltmp0-foo
	.cfi_endproc

	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# BB#0:
	pushq	%rax
.Ltmp2:
	.cfi_def_cfa_offset 16
	movq	$d, c(%rip)
	movl	$a, %eax
	testl	%eax, %eax
	sete	%al
	movzbl	%al, %esi
	movl	%esi, d(%rip)
	xorl	%edi, %edi
	callq	foo
	movl	%eax, b(%rip)
	xorl	%eax, %eax
	popq	%rdx
	ret
.Ltmp3:
	.size	main, .Ltmp3-main
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
