	.file	"add.c"
	.option nopic
	.attribute arch, "rv32i2p1_m2p0_c2p0_zmmul1p0_zca1p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	Base_add
	.type	Base_add, @function
Base_add:
	add	a0,a0,a1
	ret
	.size	Base_add, .-Base_add
	.section	.text.startup,"ax",@progbits
	.align	1
	.globl	main
	.type	main, @function
main:
	li	a0,0
	ret
	.size	main, .-main
	.ident	"GCC: (g1b306039a) 15.1.0"
	.section	.note.GNU-stack,"",@progbits
