	.file	"/home/chris/FLL/../FLL-workplace/llvmbugs/info/15920/fail.c"
	.text
	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# BB#0:
	pushq	%rax
.Ltmp1:
	.cfi_def_cfa_offset 16
	movl	$1, i(%rip)
	movb	$0, d(%rip)
	movb	$-16, c(%rip)
	movl	$.L.str, %edi
	movl	$1, %esi
	xorb	%al, %al
	callq	printf
	xorl	%eax, %eax
	popq	%rdx
	ret
.Ltmp2:
	.size	main, .Ltmp2-main
	.cfi_endproc

	.type	c,@object               # @c
	.comm	c,1,1
	.type	i,@object               # @i
	.comm	i,4,4
	.type	d,@object               # @d
	.comm	d,1,1
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	 "%d\n"
	.size	.L.str, 4


	.section	".note.GNU-stack","",@progbits
