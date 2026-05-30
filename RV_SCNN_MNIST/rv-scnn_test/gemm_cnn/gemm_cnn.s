	.file	"gemm_cnn.c"
	.option nopic
	.attribute arch, "rv32i2p1_m2p0_c2p0_zmmul1p0_zca1p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	SCNN_WB
	.type	SCNN_WB, @function
SCNN_WB:
	addi	sp,sp,-48
	sw	ra,44(sp)
	sw	s0,40(sp)
	addi	s0,sp,48
	sw	a0,-36(s0)
	lw	a5,-36(s0)
	lw	a4,-20(s0)
 #APP
# 39 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 0, 2, a5, a5, a4
# 0 "" 2
 #NO_APP
	sw	a5,-24(s0)
	nop
	lw	ra,44(sp)
	lw	s0,40(sp)
	addi	sp,sp,48
	jr	ra
	.size	SCNN_WB, .-SCNN_WB
	.align	1
	.globl	SCNN_WB_INT
	.type	SCNN_WB_INT, @function
SCNN_WB_INT:
	addi	sp,sp,-16
	sw	ra,12(sp)
	sw	s0,8(sp)
	addi	s0,sp,16
	mv	a5,a0
	mv	a0,a5
	lw	ra,12(sp)
	lw	s0,8(sp)
	addi	sp,sp,16
	tail	SCNN_WB
	.size	SCNN_WB_INT, .-SCNN_WB_INT
	.align	1
	.globl	POOL_WB
	.type	POOL_WB, @function
POOL_WB:
	addi	sp,sp,-48
	sw	ra,44(sp)
	sw	s0,40(sp)
	addi	s0,sp,48
	sw	a0,-36(s0)
	sw	zero,-20(s0)
	lw	a5,-36(s0)
	lw	a4,-20(s0)
 #APP
# 87 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 2, a5, a5, a4
# 0 "" 2
 #NO_APP
	sw	a5,-24(s0)
	nop
	lw	ra,44(sp)
	lw	s0,40(sp)
	addi	sp,sp,48
	jr	ra
	.size	POOL_WB, .-POOL_WB
	.align	1
	.globl	POOL_WB_INT
	.type	POOL_WB_INT, @function
POOL_WB_INT:
	addi	sp,sp,-48
	sw	ra,44(sp)
	sw	s0,40(sp)
	addi	s0,sp,48
	sw	a0,-36(s0)
	sw	a1,-40(s0)
	lw	a5,-36(s0)
	lw	a4,-40(s0)
 #APP
# 98 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 2, a5, a5, a4
# 0 "" 2
 #NO_APP
	sw	a5,-20(s0)
	nop
	lw	ra,44(sp)
	lw	s0,40(sp)
	addi	sp,sp,48
	jr	ra
	.size	POOL_WB_INT, .-POOL_WB_INT
	.align	1
	.globl	L_MODE
	.type	L_MODE, @function
L_MODE:
	slli	a1,a1,1
	add	a1,a1,a0
	slli	a3,a3,4
	add	a1,a1,a3
 #APP
# 6 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 4, a1, a1, a2
# 0 "" 2
 #NO_APP
	ret
	.size	L_MODE, .-L_MODE
	.align	1
	.globl	L_SCNN
	.type	L_SCNN, @function
L_SCNN:
	slli	a1,a1,16
	slli	a3,a3,16
	add	a3,a3,a0
	add	a1,a1,a2
 #APP
# 18 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 0, 0, a3, a3, a1
# 0 "" 2
 #NO_APP
	ret
	.size	L_SCNN, .-L_SCNN
	.align	1
	.globl	SCNN4x4
	.type	SCNN4x4, @function
SCNN4x4:
 #APP
# 28 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 0, 1, a0, a0, a1
# 0 "" 2
 #NO_APP
	ret
	.size	SCNN4x4, .-SCNN4x4
	.align	1
	.globl	POOL_RI
	.type	POOL_RI, @function
POOL_RI:
	li	a5,0
 #APP
# 63 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 0, a0, a0, a5
# 0 "" 2
 #NO_APP
	ret
	.size	POOL_RI, .-POOL_RI
	.align	1
	.globl	POOL
	.type	POOL, @function
POOL:
	li	a5,0
 #APP
# 75 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 1, a0, a0, a5
# 0 "" 2
 #NO_APP
	ret
	.size	POOL, .-POOL
	.align	1
	.globl	RELU
	.type	RELU, @function
RELU:
 #APP
# 123 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 3, 0, a0, a0, a1
# 0 "" 2
 #NO_APP
	ret
	.size	RELU, .-RELU
	.align	1
	.globl	record
	.type	record, @function
record:
 #APP
# 136 "../include/custom.h" 1
	rdcycle a0
# 0 "" 2
 #NO_APP
	ret
	.size	record, .-record
	.align	1
	.globl	Base_GEMM
	.type	Base_GEMM, @function
Base_GEMM:
	addi	sp,sp,-32
	sw	s2,20(sp)
	mv	s2,a0
	mul	a0,a0,a1
	sw	s1,24(sp)
	mv	s1,a1
	li	a1,4
	sw	ra,28(sp)
	sw	a2,12(sp)
	sw	a3,8(sp)
	sw	a4,4(sp)
	call	calloc
	ble	s2,zero,.L13
	ble	s1,zero,.L13
	lw	a2,12(sp)
	ble	a2,zero,.L13
	lw	t5,8(sp)
	lw	a4,4(sp)
	li	t0,0
	li	t6,0
.L15:
	slli	t3,t0,2
	add	t3,a0,t3
	li	t4,0
.L18:
	lw	a7,0(t3)
	add	a6,a4,t4
	mv	a1,t5
	li	a3,0
.L16:
	lb	a5,0(a1)
	lb	t1,0(a6)
	addi	a3,a3,1
	add	a1,a1,s2
	mul	a5,a5,t1
	add	a6,a6,s1
	add	a7,a7,a5
	bne	a2,a3,.L16
	sw	a7,0(t3)
	addi	t4,t4,1
	addi	t3,t3,4
	bne	s1,t4,.L18
	addi	t6,t6,1
	addi	t5,t5,1
	add	t0,t0,s1
	bne	s2,t6,.L15
.L13:
	lw	ra,28(sp)
	lw	s1,24(sp)
	lw	s2,20(sp)
	addi	sp,sp,32
	jr	ra
	.size	Base_GEMM, .-Base_GEMM
	.align	1
	.globl	error_check
	.type	error_check, @function
error_check:
	ble	a2,zero,.L24
	slli	a2,a2,2
	mv	a4,a0
	add	a2,a0,a2
	li	a0,0
.L23:
	lw	a3,0(a1)
	lw	a5,0(a4)
	addi	a4,a4,4
	addi	a1,a1,4
	sub	a5,a5,a3
	srai	a3,a5,31
	xor	a5,a3,a5
	sub	a5,a5,a3
	add	a0,a0,a5
	bne	a2,a4,.L23
	ret
.L24:
	li	a0,0
	ret
	.size	error_check, .-error_check
	.section	.text.startup,"ax",@progbits
	.align	1
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-80
	li	a0,192
	sw	ra,76(sp)
	sw	s1,72(sp)
	sw	s2,68(sp)
	sw	s3,64(sp)
	sw	s4,60(sp)
	sw	s5,56(sp)
	sw	s6,52(sp)
	sw	s7,48(sp)
	sw	s8,44(sp)
	sw	s9,40(sp)
	sw	s10,36(sp)
	sw	s11,32(sp)
	call	malloc
	mv	s2,a0
	li	a0,192
	call	malloc
	li	a1,-858992640
	mv	s1,a0
	addi	a1,a1,-819
	li	a4,0
	li	a0,192
.L27:
	mulhu	a3,a4,a1
	add	a2,s2,a4
	srli	a3,a3,4
	slli	a5,a3,2
	add	a5,a5,a3
	slli	a5,a5,2
	sub	a5,a4,a5
	addi	a5,a5,-7
	sb	a5,0(a2)
	addi	a4,a4,1
	bne	a4,a0,.L27
	li	a1,128208896
	addi	a1,a1,-917
	li	a4,0
	li	a0,192
.L28:
	mulhu	a3,a4,a1
	add	a2,s1,a4
	srli	a3,a3,1
	slli	a5,a3,4
	add	a5,a5,a3
	slli	a5,a5,2
	sub	a5,a5,a3
	sub	a5,a4,a5
	addi	a5,a5,-5
	sb	a5,0(a2)
	addi	a4,a4,1
	bne	a4,a0,.L28
	li	a0,144
	call	malloc
	mv	s6,a0
	sw	a0,16(sp)
 #APP
# 136 "../include/custom.h" 1
	rdcycle s4
# 0 "" 2
 #NO_APP
	li	a5,393216
	addi	a5,a5,6
	li	s3,32
 #APP
# 18 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 0, 0, a5, s3, a5
# 0 "" 2
 #NO_APP
	li	a5,2
	li	a4,0
 #APP
# 6 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 4, a5, a5, a4
# 0 "" 2
# 28 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 0, 1, a5, s2, s1
# 0 "" 2
 #NO_APP
	call	SCNN_WB
 #APP
# 136 "../include/custom.h" 1
	rdcycle s5
# 0 "" 2
 #NO_APP
	li	a1,6
	mv	a4,s1
	mv	a3,s2
	mv	a0,a1
	mv	a2,s3
	call	Base_GEMM
	lw	a4,16(sp)
	mv	a3,a0
	sw	a0,24(sp)
	addi	a1,s6,144
	li	s8,0
.L29:
	lw	a2,0(a3)
	lw	a5,0(a4)
	addi	a4,a4,4
	addi	a3,a3,4
	sub	a5,a5,a2
	srai	a2,a5,31
	xor	a5,a2,a5
	sub	a5,a5,a2
	add	s8,s8,a5
	bne	a1,a4,.L29
	sub	a5,s5,s4
	mv	a0,a5
	sw	a5,20(sp)
	lui	s10,%hi(.LANCHOR0)
	call	__floatsidf
	addi	a5,s10,%lo(.LANCHOR0)
	mv	a2,a0
	mv	a3,a1
	lw	a0,0(a5)
	lw	a1,4(a5)
	sw	a5,4(sp)
	lui	s5,%hi(.LC4)
	call	__divdf3
	lw	a5,4(sp)
	addi	s5,s5,%lo(.LC4)
	li	s7,24
	lw	a2,8(a5)
	lw	a3,12(a5)
	li	s3,4
	call	__muldf3
	call	__truncdfsf2
	lw	a5,4(sp)
	sw	a0,28(sp)
	addi	a0,a5,16
	call	puts
	lui	a5,%hi(.LANCHOR0+36)
	addi	a5,a5,%lo(.LANCHOR0+36)
	sw	a5,12(sp)
	sw	zero,0(sp)
.L30:
	lw	s6,16(sp)
	addi	a5,s7,-24
	li	s9,0
	sw	a5,8(sp)
.L34:
	lw	a1,0(sp)
	lw	a0,12(sp)
	mv	a2,s9
	call	printf
	lw	s4,8(sp)
.L31:
	slli	a5,s4,2
	add	s10,a5,s6
	li	s11,0
.L32:
	lw	a1,0(s10)
	mv	a0,s5
	addi	s11,s11,1
	call	printf
	addi	s10,s10,4
	bne	s11,s3,.L32
	li	a0,10
	addi	s4,s4,6
	call	putchar
	bne	s7,s4,.L31
	li	a0,10
	call	putchar
	addi	s6,s6,16
	beq	s9,zero,.L36
	lw	a5,0(sp)
	addi	s7,s7,24
	bne	a5,zero,.L35
	sw	s3,0(sp)
	j	.L30
.L36:
	mv	s9,s3
	j	.L34
.L35:
	lw	a5,4(sp)
	li	a2,6
	mv	a1,a2
	li	a3,32
	addi	a0,a5,72
	call	printf
	lw	a5,4(sp)
	mv	a1,s8
	addi	a0,a5,88
	call	printf
	lw	a5,4(sp)
	lw	a1,20(sp)
	addi	a0,a5,100
	call	printf
	lw	a0,28(sp)
	call	__extendsfdf2
	lw	a5,4(sp)
	mv	a2,a0
	mv	a3,a1
	addi	a0,a5,116
	call	printf
	mv	a0,s2
	call	free
	mv	a0,s1
	call	free
	lw	a0,16(sp)
	call	free
	lw	a0,24(sp)
	call	free
	lw	ra,76(sp)
	lw	s1,72(sp)
	lw	s2,68(sp)
	lw	s3,64(sp)
	lw	s4,60(sp)
	lw	s5,56(sp)
	lw	s6,52(sp)
	lw	s7,48(sp)
	lw	s8,44(sp)
	lw	s9,40(sp)
	lw	s10,36(sp)
	lw	s11,32(sp)
	li	a0,0
	addi	sp,sp,80
	jr	ra
	.size	main, .-main
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.word	0
	.word	1083310080
.LC1:
	.word	858993459
	.word	1070805811
.LC2:
	.string	"Accelerator Output:"
.LC3:
	.string	"Block starting at (%d,%d):\n"
.LC4:
	.string	"%5d "
	.zero	3
.LC5:
	.string	"M:%d N:%d K:%d\n"
.LC6:
	.string	"error:%d\n"
	.zero	2
.LC7:
	.string	"cycle count:%d\n"
.LC8:
	.string	"GOPS:%.4f\n"
	.globl	__extendsfdf2
	.globl	__truncdfsf2
	.globl	__muldf3
	.globl	__divdf3
	.globl	__floatsidf
	.ident	"GCC: (g1b306039a) 15.1.0"
	.section	.note.GNU-stack,"",@progbits
