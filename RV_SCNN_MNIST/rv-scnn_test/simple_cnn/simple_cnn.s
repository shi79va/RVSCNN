	.file	"simple_cnn.c"
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
	.globl	record
	.type	record, @function
record:
 #APP
# 109 "../include/custom.h" 1
	rdcycle a0
# 0 "" 2
 #NO_APP
	ret
	.size	record, .-record
	.align	1
	.globl	Base_conv
	.type	Base_conv, @function
Base_conv:
	addi	sp,sp,-112
	sw	s7,80(sp)
	sw	s9,72(sp)
	slli	s7,a6,1
	mv	s9,a0
	addi	a0,a0,-3
	add	a0,a0,s7
	sw	s10,68(sp)
	mv	s10,a5
	div	a5,a0,a5
	sw	s11,64(sp)
	sw	s6,84(sp)
	add	s7,s7,s9
	sw	s1,104(sp)
	sw	s3,96(sp)
	mv	s3,a1
	li	a1,4
	sw	ra,108(sp)
	sw	s2,100(sp)
	sw	s4,92(sp)
	sw	s5,88(sp)
	sw	s8,76(sp)
	mv	s4,a6
	mv	s2,a2
	mv	s8,a3
	sw	a4,8(sp)
	addi	s11,a5,1
	mul	s6,s11,s11
	sw	a5,16(sp)
	mul	a0,s6,a2
	mul	s1,s7,s7
	call	calloc
	sw	a0,20(sp)
	li	a1,1
	mul	a0,s1,s3
	call	calloc
	mv	s5,a0
	ble	s3,zero,.L13
	ble	s9,zero,.L13
	mul	a5,s4,s7
	sw	s10,32(sp)
	sw	s6,12(sp)
	sw	s11,24(sp)
	sw	s2,28(sp)
	mv	s11,s3
	sw	a0,36(sp)
	li	s3,0
	mv	s5,s1
	mul	s10,s9,s9
	add	a5,a5,s4
	add	s6,a0,a5
	li	s4,0
.L15:
	add	a5,s6,s4
	mv	s2,s8
	li	s1,0
.L16:
	mv	a1,s2
	mv	a0,a5
	mv	a2,s9
	call	memcpy
	addi	s1,s1,1
	add	a5,a0,s7
	add	s2,s2,s9
	bne	s9,s1,.L16
	addi	s3,s3,1
	add	s8,s8,s10
	add	s4,s4,s5
	bne	s11,s3,.L15
	mv	s1,s5
	mv	s3,s11
	lw	s6,12(sp)
	lw	s11,24(sp)
	lw	s2,28(sp)
	lw	s10,32(sp)
	lw	s5,36(sp)
.L13:
	ble	s2,zero,.L17
	ble	s11,zero,.L17
	mul	a0,s10,s7
	lw	s9,8(sp)
	lw	a4,20(sp)
	slli	a5,s2,1
	slli	t3,s2,3
	slli	a2,s2,2
	add	s8,s9,a5
	mv	a3,s11
	add	t5,a2,s2
	sub	t4,t3,s2
	add	t1,t3,s2
	slli	a6,s7,1
	li	a5,0
	li	a1,0
	addi	a7,a4,4
	mv	s11,s10
.L19:
	lw	t6,16(sp)
	sw	s6,24(sp)
	sw	a1,28(sp)
	mv	s4,s5
	mv	a1,a3
	mv	a4,a5
	li	a3,0
	mv	s6,s5
.L26:
	add	t2,t6,a4
	lw	t6,20(sp)
	slli	t2,t2,2
	slli	t0,a4,2
	add	t0,t6,t0
	add	t2,a7,t2
	add	s5,a6,s4
	li	t6,0
	add	s10,s7,s4
	ble	s3,zero,.L23
	sw	s7,36(sp)
	sw	a1,32(sp)
	sw	a3,40(sp)
	sw	a5,44(sp)
	sw	a6,48(sp)
	sw	a0,52(sp)
	sw	a4,56(sp)
	sw	a7,60(sp)
	mv	s7,s6
.L21:
	add	a6,t6,s4
	add	a0,s10,t6
	add	a1,t6,s5
	mv	a3,s8
	mv	a4,s9
	li	a7,0
	li	a5,0
	sw	t0,8(sp)
	sw	t6,12(sp)
.L20:
	lb	s6,0(a6)
	lb	t0,0(a4)
	add	t6,a4,s2
	lb	t6,0(t6)
	mul	t0,t0,s6
	lb	s6,1(a6)
	addi	a7,a7,1
	mul	t6,t6,s6
	add	a5,t0,a5
	lb	s6,2(a6)
	lb	t0,0(a3)
	add	a6,a6,s1
	mul	t0,t0,s6
	add	a5,t6,a5
	add	t6,a3,s2
	lb	s6,0(a0)
	lb	t6,0(t6)
	add	a5,t0,a5
	add	t0,a4,a2
	mul	t6,t6,s6
	lb	t0,0(t0)
	lb	s6,1(a0)
	mul	t0,t0,s6
	add	a5,t6,a5
	add	t6,a4,t5
	lb	s6,2(a0)
	lb	t6,0(t6)
	add	a0,a0,s1
	add	a5,t0,a5
	add	t0,a3,a2
	mul	t6,t6,s6
	lb	t0,0(t0)
	lb	s6,0(a1)
	add	a3,a3,t1
	mul	t0,t0,s6
	add	a5,t6,a5
	add	t6,a4,t4
	lb	s6,1(a1)
	lb	t6,0(t6)
	add	t0,t0,a5
	add	a5,a4,t3
	mul	t6,t6,s6
	lb	a5,0(a5)
	lb	s6,2(a1)
	add	a4,a4,t1
	add	a1,a1,s1
	mul	a5,a5,s6
	add	t6,t6,t0
	add	a5,a5,t6
	bne	s3,a7,.L20
	lw	t0,8(sp)
	lw	t6,12(sp)
	sw	a5,0(t0)
	addi	t0,t0,4
	add	t6,t6,s11
	bne	t0,t2,.L21
	mv	s6,s7
	lw	a1,32(sp)
	lw	a3,40(sp)
	lw	a5,44(sp)
	lw	a6,48(sp)
	lw	a0,52(sp)
	lw	a4,56(sp)
	lw	a7,60(sp)
	lw	s7,36(sp)
.L22:
	lw	t6,16(sp)
	add	a4,a4,a1
	add	s4,s4,a0
	beq	t6,a3,.L24
	addi	a3,a3,1
	j	.L26
.L23:
	sw	zero,0(t0)
	addi	t0,t0,4
	beq	t2,t0,.L22
	sw	zero,0(t0)
	addi	t0,t0,4
	bne	t2,t0,.L23
	j	.L22
.L24:
	mv	a3,a1
	lw	a1,28(sp)
	mv	s5,s6
	lw	s6,24(sp)
	addi	a1,a1,1
	addi	s9,s9,1
	add	a5,a5,s6
	addi	s8,s8,1
	bne	s2,a1,.L19
.L17:
	mv	a0,s5
	call	free
	lw	ra,108(sp)
	lw	a0,20(sp)
	lw	s1,104(sp)
	lw	s2,100(sp)
	lw	s3,96(sp)
	lw	s4,92(sp)
	lw	s5,88(sp)
	lw	s6,84(sp)
	lw	s7,80(sp)
	lw	s8,76(sp)
	lw	s9,72(sp)
	lw	s10,68(sp)
	lw	s11,64(sp)
	addi	sp,sp,112
	jr	ra
	.size	Base_conv, .-Base_conv
	.align	1
	.globl	MaxPool2x2
	.type	MaxPool2x2, @function
MaxPool2x2:
	addi	sp,sp,-64
	addi	a5,a2,1
	sw	s1,56(sp)
	srli	s1,a5,31
	add	s1,s1,a5
	srai	s1,s1,1
	sw	s7,32(sp)
	mul	s7,s1,s1
	sw	s6,36(sp)
	mv	s6,a1
	sw	s5,40(sp)
	mv	s5,a0
	li	a1,4
	sw	ra,60(sp)
	sw	a2,8(sp)
	mul	a0,s7,s6
	call	calloc
	mv	a6,a0
	ble	s6,zero,.L39
	lw	a2,8(sp)
	ble	a2,zero,.L39
	mul	a7,a2,a2
	sw	s2,52(sp)
	sw	s3,48(sp)
	sw	s4,44(sp)
	sw	s8,28(sp)
	sw	s9,24(sp)
	sw	s10,20(sp)
	sw	s11,16(sp)
	slli	s3,a2,1
	slli	s2,s1,2
	slli	a0,a2,3
	slli	s4,a2,2
	li	a4,0
	li	a1,0
	li	a3,0
	sw	s7,12(sp)
.L42:
	slli	s8,a1,2
	slli	s7,a4,2
	add	s8,a6,s8
	add	s7,s5,s7
	add	s10,a2,a4
	li	s9,1
	li	t0,0
	mv	t2,a3
.L59:
	slli	s11,t0,1
	ble	a2,s9,.L61
	add	t4,s7,s4
	mv	t3,s7
	mv	t6,s8
	li	a3,0
	li	t5,0
	sw	s9,8(sp)
.L51:
	mv	a5,a3
	bge	a3,s11,.L44
	mv	a5,s11
.L44:
	ble	a2,a5,.L60
	lw	t1,0(t3)
	addi	a5,a3,1
	not	s9,t1
	srai	s9,s9,31
	and	t1,t1,s9
	bge	a5,a2,.L46
	lw	s9,4(t3)
	lw	a5,0(t4)
	bge	a5,s9,.L47
	mv	a5,s9
.L47:
	lw	s9,4(t4)
	bge	a5,s9,.L70
	mv	a5,s9
.L70:
	bge	a5,t1,.L43
	mv	a5,t1
.L43:
	sw	a5,0(t6)
	addi	t5,t5,1
	addi	a3,a3,2
	addi	t6,t6,4
	addi	t3,t3,8
	addi	t4,t4,8
	bgt	s1,t5,.L51
	lw	s9,8(sp)
.L58:
	addi	t0,t0,1
	addi	s9,s9,2
	add	s10,s10,s3
	add	s8,s8,s2
	add	s7,s7,a0
	bgt	s1,t0,.L59
	lw	a5,12(sp)
	addi	a3,t2,1
	add	a4,a4,a7
	add	a1,a1,a5
	bne	s6,a3,.L42
	lw	s2,52(sp)
	lw	s3,48(sp)
	lw	s4,44(sp)
	lw	s8,28(sp)
	lw	s9,24(sp)
	lw	s10,20(sp)
	lw	s11,16(sp)
.L39:
	lw	ra,60(sp)
	lw	s1,56(sp)
	lw	s5,40(sp)
	lw	s6,36(sp)
	lw	s7,32(sp)
	mv	a0,a6
	addi	sp,sp,64
	jr	ra
.L60:
	li	a5,0
	j	.L43
.L46:
	sub	a5,s5,s7
	add	a5,a5,t3
	slli	s9,s10,2
	add	a5,a5,s9
	lw	a5,0(a5)
	j	.L70
.L61:
	mv	t1,s7
	mv	t5,s8
	li	a3,0
	li	t4,0
.L53:
	addi	t6,a3,1
	mv	a5,a3
	bge	a3,s11,.L55
	mv	a5,s11
.L55:
	ble	a2,a5,.L62
	lw	a5,0(t1)
	not	t3,a5
	srai	t3,t3,31
	and	a5,a5,t3
	ble	a2,t6,.L54
	lw	t3,4(t1)
	bge	a5,t3,.L54
	mv	a5,t3
.L54:
	sw	a5,0(t5)
	addi	t4,t4,1
	addi	a3,a3,2
	addi	t5,t5,4
	addi	t1,t1,8
	bgt	s1,t4,.L53
	j	.L58
.L62:
	li	a5,0
	j	.L54
	.size	MaxPool2x2, .-MaxPool2x2
	.align	1
	.globl	RVSCNN_conv_mp
	.type	RVSCNN_conv_mp, @function
RVSCNN_conv_mp:
	addi	sp,sp,-240
	sw	s11,192(sp)
	sw	a6,88(sp)
	mv	s11,a0
	slli	a6,a6,1
	addi	a0,a0,-3
	add	a0,a0,a6
	sw	s1,232(sp)
	mv	s1,a5
	div	a5,a0,a5
	sw	s2,228(sp)
	mv	s2,a4
	sw	s8,204(sp)
	sw	a1,104(sp)
	li	a1,4
	sw	s10,196(sp)
	sw	ra,236(sp)
	sw	a2,132(sp)
	mv	s10,a3
	addi	a4,a5,2
	sw	a5,112(sp)
	srli	a5,a4,31
	add	a5,a5,a4
	srai	a4,a5,1
	mul	s8,a4,a4
	sw	a4,24(sp)
	mul	a0,s8,a2
	call	calloc
	sw	a0,120(sp)
	beq	a0,zero,.L226
	lui	a5,%hi(.LANCHOR0)
	addi	a5,a5,%lo(.LANCHOR0)
	sw	a5,108(sp)
	addi	a0,a5,16
	lw	a5,112(sp)
	sw	s3,224(sp)
	sw	s4,220(sp)
	addi	a5,a5,1
	sw	a5,124(sp)
	lw	a5,132(sp)
	sw	zero,172(sp)
	srai	a5,a5,2
	sw	a5,116(sp)
	call	puts
	lw	a4,116(sp)
	slli	a5,s11,16
	add	a5,a5,s8
	ble	a4,zero,.L74
	lw	a4,104(sp)
	lw	a3,104(sp)
	sw	s8,20(sp)
	slli	a4,a4,3
	add	a4,a4,a3
	slli	a4,a4,2
	sw	a4,128(sp)
	lui	a3,%hi(.LANCHOR0+32)
	lw	s8,128(sp)
	mv	a2,s2
	sw	s5,216(sp)
	sw	s6,212(sp)
	sw	s7,208(sp)
	lui	s6,%hi(.LC3)
	mv	s7,s2
	addi	a4,a3,%lo(.LANCHOR0+32)
	li	s5,-1431654400
	li	s4,1431654400
	li	s3,954437632
	li	s2,477216768
	sw	s1,28(sp)
	sw	s9,200(sp)
	sw	a4,8(sp)
	addi	s6,s6,%lo(.LC3)
	addi	s5,s5,-1365
	addi	s4,s4,1365
	addi	s3,s3,-455
	addi	s2,s2,1820
	sw	a2,12(sp)
	sw	a5,16(sp)
	li	s1,0
.L76:
	lw	a0,8(sp)
	mv	a1,s1
	li	s9,0
	call	printf
	lw	a5,104(sp)
	ble	a5,zero,.L80
.L75:
	add	a5,s7,s9
	lb	a1,0(a5)
	mv	a0,s6
	addi	s9,s9,1
	call	printf
	mul	a5,s9,s5
	bleu	a5,s4,.L227
.L78:
	mul	a5,s9,s3
	bleu	a5,s2,.L228
	blt	s9,s8,.L75
.L80:
	lw	a5,116(sp)
	addi	s1,s1,1
	add	s7,s7,s8
	bne	a5,s1,.L76
	lw	a2,12(sp)
	lw	a5,16(sp)
	lw	s8,20(sp)
	lw	s1,28(sp)
 #APP
# 109 "../include/custom.h" 1
	rdcycle s4
# 0 "" 2
 #NO_APP
	lw	a4,104(sp)
 #APP
# 18 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 0, 0, a5, a4, a5
# 0 "" 2
 #NO_APP
	lw	a5,88(sp)
	li	s3,3
	slli	s9,s1,1
	sub	s3,s3,a5
	mul	s3,s3,s11
	sub	a5,s10,a5
	slli	a4,s8,3
	sw	a4,52(sp)
	lw	a4,120(sp)
	slli	s10,s8,2
	lui	a1,%hi(.LANCHOR0+52)
	add	s6,a4,s10
	lui	a3,%hi(.LANCHOR0+72)
	addi	a4,a1,%lo(.LANCHOR0+52)
	add	s3,s3,a5
	mul	a5,s11,s9
	sw	a4,68(sp)
	addi	a4,a3,%lo(.LANCHOR0+72)
	sw	a4,72(sp)
	mv	a3,s10
	li	s2,0
	mv	s10,s9
	sw	s4,136(sp)
	mv	s9,s11
	li	a7,0
	sw	a5,100(sp)
	mv	s11,s6
.L117:
	li	a4,18
	li	a5,0
 #APP
# 6 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 4, a4, a4, a5
# 0 "" 2
 #NO_APP
	li	a4,20
 #APP
# 6 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 4, a4, a4, a5
# 0 "" 2
 #NO_APP
	li	a4,22
 #APP
# 6 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 4, a4, a4, a5
# 0 "" 2
 #NO_APP
	li	a4,24
 #APP
# 6 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 4, a4, a4, a5
# 0 "" 2
 #NO_APP
	lw	a5,112(sp)
	blt	a5,zero,.L82
	lw	s1,120(sp)
	slli	a5,s2,2
	lui	a0,%hi(.LANCHOR0+100)
	add	s1,s1,a5
	add	a5,s11,a5
	lui	a1,%hi(.LANCHOR0+128)
	sw	a5,92(sp)
	addi	a5,a0,%lo(.LANCHOR0+100)
	lui	a4,%hi(.LANCHOR0+156)
	sw	a5,56(sp)
	addi	a5,a1,%lo(.LANCHOR0+128)
	sw	a5,60(sp)
	addi	a5,a4,%lo(.LANCHOR0+156)
	sw	a5,64(sp)
	lw	a5,88(sp)
	lui	a6,%hi(.LC3)
	sw	s1,96(sp)
	neg	s1,a5
	addi	s8,a6,%lo(.LC3)
	sw	s3,80(sp)
	sw	zero,84(sp)
	li	s4,0
	li	s6,16
	sw	s2,140(sp)
	sw	s11,144(sp)
	sw	a7,148(sp)
	sw	s3,152(sp)
	sw	a3,156(sp)
	sw	s1,76(sp)
	sw	a2,48(sp)
.L83:
	lw	a5,76(sp)
	lw	a2,76(sp)
	lw	s1,80(sp)
	addi	a3,a5,1
	not	a6,a3
	addi	a4,a5,2
	srli	a6,a6,31
	slt	a3,a3,s9
	not	a0,a4
	and	a3,a3,a6
	addi	a5,a5,3
	sw	a3,36(sp)
	slt	a4,a4,s9
	srli	a3,a0,31
	not	a1,a5
	and	a4,a4,a3
	sw	a4,40(sp)
	slt	a5,a5,s9
	srli	a4,a1,31
	and	a5,a5,a4
	lw	a4,76(sp)
	sw	a5,44(sp)
	not	a5,a2
	sub	s7,s1,s9
	sgt	a4,s9,a4
	srli	a5,a5,31
	and	a5,a4,a5
	sub	s5,s7,s9
	sub	a4,s5,s9
	sw	a5,32(sp)
	lw	a5,84(sp)
	sw	a4,16(sp)
	lw	a4,96(sp)
	slli	a5,a5,2
	sw	s1,20(sp)
	add	a4,a4,a5
	sw	a4,12(sp)
	lw	a4,92(sp)
	add	a5,a4,a5
	sw	a5,8(sp)
	lw	a5,88(sp)
	neg	s11,a5
	mv	s1,s11
	mv	s11,s7
	mv	s7,s5
	li	s5,0
.L116:
	lw	a5,32(sp)
	sw	zero,176(sp)
	sw	zero,180(sp)
	sw	zero,184(sp)
	sw	zero,188(sp)
	beq	a5,zero,.L85
	blt	s1,zero,.L86
	ble	s9,s1,.L86
	lw	a5,16(sp)
	lbu	a4,0(a5)
	sb	a4,176(sp)
.L86:
	addi	a4,s1,1
	blt	a4,zero,.L87
	ble	s9,a4,.L87
	lw	a5,16(sp)
	lbu	a4,1(a5)
	sb	a4,177(sp)
.L87:
	addi	a4,s1,2
	blt	a4,zero,.L88
	ble	s9,a4,.L88
	lw	a5,16(sp)
	lbu	a4,2(a5)
	sb	a4,178(sp)
.L88:
	addi	a4,s1,3
	blt	a4,zero,.L85
	ble	s9,a4,.L85
	lw	a5,16(sp)
	lbu	a4,3(a5)
	sb	a4,179(sp)
.L85:
	lw	a5,36(sp)
	beq	a5,zero,.L91
	blt	s1,zero,.L92
	ble	s9,s1,.L92
	lbu	a4,0(s7)
	sb	a4,180(sp)
.L92:
	addi	a4,s1,1
	blt	a4,zero,.L93
	ble	s9,a4,.L93
	lbu	a4,1(s7)
	sb	a4,181(sp)
.L93:
	addi	a4,s1,2
	blt	a4,zero,.L94
	ble	s9,a4,.L94
	lbu	a4,2(s7)
	sb	a4,182(sp)
.L94:
	addi	a4,s1,3
	blt	a4,zero,.L91
	ble	s9,a4,.L91
	lbu	a4,3(s7)
	sb	a4,183(sp)
.L91:
	lw	a5,40(sp)
	beq	a5,zero,.L97
	blt	s1,zero,.L98
	ble	s9,s1,.L98
	lbu	a4,0(s11)
	sb	a4,184(sp)
.L98:
	addi	a4,s1,1
	blt	a4,zero,.L99
	ble	s9,a4,.L99
	lbu	a4,1(s11)
	sb	a4,185(sp)
.L99:
	addi	a4,s1,2
	blt	a4,zero,.L100
	ble	s9,a4,.L100
	lbu	a4,2(s11)
	sb	a4,186(sp)
.L100:
	addi	a4,s1,3
	blt	a4,zero,.L97
	ble	s9,a4,.L97
	lbu	a4,3(s11)
	sb	a4,187(sp)
.L97:
	lw	a5,44(sp)
	beq	a5,zero,.L103
	blt	s1,zero,.L104
	ble	s9,s1,.L104
	lw	a5,20(sp)
	lbu	a4,0(a5)
	sb	a4,188(sp)
.L104:
	addi	a4,s1,1
	blt	a4,zero,.L105
	ble	s9,a4,.L105
	lw	a5,20(sp)
	lbu	a4,1(a5)
	sb	a4,189(sp)
.L105:
	addi	a4,s1,2
	blt	a4,zero,.L106
	ble	s9,a4,.L106
	lw	a5,20(sp)
	lbu	a4,2(a5)
	sb	a4,190(sp)
.L106:
	addi	a4,s1,3
	blt	a4,zero,.L103
	ble	s9,a4,.L103
	lw	a5,20(sp)
	lbu	a4,3(a5)
	sb	a4,191(sp)
.L103:
	lw	a0,68(sp)
	mv	a2,s5
	mv	a1,s4
	call	printf
	addi	s2,sp,180
	li	s3,0
	sw	s1,28(sp)
.L108:
	addi	s1,s2,-4
.L109:
	lb	a1,0(s1)
	mv	a0,s8
	addi	s1,s1,1
	call	printf
	bne	s2,s1,.L109
	li	a0,10
	addi	s3,s3,4
	call	putchar
	addi	s2,s2,4
	bne	s3,s6,.L108
	lw	s1,28(sp)
	lw	a5,48(sp)
	addi	a4,sp,176
 #APP
# 28 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 0, 1, a4, a5, a4
# 0 "" 2
 #NO_APP
	li	a4,0
	li	a5,2
 #APP
# 75 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 2, 1, a4, a5, a4
# 0 "" 2
 #NO_APP
	li	a1,1
	addi	a0,sp,172
	call	POOL_WB_INT
	lw	a1,172(sp)
	lw	a0,72(sp)
	call	printf
	mv	a4,s4
	bge	s4,s5,.L112
	mv	a4,s5
.L112:
	lw	a5,24(sp)
	bgt	a5,a4,.L111
	li	a1,2
	addi	a0,sp,172
	call	POOL_WB_INT
	lw	a1,172(sp)
	lw	a0,56(sp)
	addi	s5,s5,1
	call	printf
	addi	a0,sp,172
	li	a1,3
	call	POOL_WB_INT
	lw	a1,172(sp)
	lw	a0,60(sp)
	call	printf
	addi	a0,sp,172
	li	a1,4
	call	POOL_WB_INT
	lw	a1,172(sp)
	lw	a0,64(sp)
	call	printf
	lw	a5,24(sp)
	bge	s5,a5,.L114
.L113:
	lw	a5,12(sp)
	add	s1,s1,s10
	add	s11,s11,s10
	addi	a5,a5,4
	sw	a5,12(sp)
	lw	a5,8(sp)
	add	s7,s7,s10
	addi	a5,a5,4
	sw	a5,8(sp)
	lw	a5,20(sp)
	add	a5,a5,s10
	sw	a5,20(sp)
	lw	a5,16(sp)
	add	a5,a5,s10
	sw	a5,16(sp)
	j	.L116
.L228:
	li	a0,10
	call	putchar
	blt	s9,s8,.L75
	j	.L80
.L111:
	lw	a4,172(sp)
	lw	a5,12(sp)
	li	a1,2
	addi	a0,sp,172
	sw	a4,0(a5)
	call	POOL_WB_INT
	lw	a1,172(sp)
	lw	a0,56(sp)
	addi	s5,s5,1
	call	printf
	lw	a4,172(sp)
	lw	a5,8(sp)
	addi	a0,sp,172
	li	a1,3
	sw	a4,0(a5)
	call	POOL_WB_INT
	lw	a1,172(sp)
	lw	a0,60(sp)
	call	printf
	lw	a5,12(sp)
	lw	a4,52(sp)
	lw	a3,172(sp)
	addi	a0,sp,172
	add	a4,a5,a4
	sw	a3,0(a4)
	li	a1,4
	call	POOL_WB_INT
	lw	a1,172(sp)
	lw	a0,64(sp)
	call	printf
	lw	a5,8(sp)
	lw	a4,52(sp)
	lw	a3,172(sp)
	add	a4,a5,a4
	lw	a5,24(sp)
	sw	a3,0(a4)
	blt	s5,a5,.L113
.L114:
	lw	a5,76(sp)
	lw	a4,24(sp)
	addi	s4,s4,1
	add	a5,a5,s10
	sw	a5,76(sp)
	lw	a5,84(sp)
	add	a5,a5,a4
	sw	a5,84(sp)
	lw	a4,100(sp)
	lw	a5,80(sp)
	add	a5,a5,a4
	sw	a5,80(sp)
	lw	a5,24(sp)
	blt	s4,a5,.L83
	lw	s2,140(sp)
	lw	s11,144(sp)
	lw	a7,148(sp)
	lw	s3,152(sp)
	lw	a3,156(sp)
	lw	a2,48(sp)
.L82:
	lw	a5,128(sp)
	addi	a7,a7,1
	add	s2,s2,a3
	add	a2,a2,a5
	lw	a5,116(sp)
	bne	a5,a7,.L117
	lw	s4,136(sp)
	lw	s5,216(sp)
	lw	s6,212(sp)
	lw	s7,208(sp)
	lw	s9,200(sp)
.L81:
 #APP
# 109 "../include/custom.h" 1
	rdcycle s1
# 0 "" 2
 #NO_APP
	lw	a5,108(sp)
	sub	a1,s1,s4
	addi	a0,a5,184
	call	printf
	lw	a4,132(sp)
	lw	a5,124(sp)
	mul	a5,a5,a4
	lw	a4,124(sp)
	mul	a5,a5,a4
	lw	a4,104(sp)
	mul	a5,a5,a4
	slli	a0,a5,3
	add	a0,a0,a5
	call	__floatsidf
	mv	s2,a0
	sub	a0,s1,s4
	mv	s3,a1
	call	__floatsidf
	mv	a2,a0
	mv	a3,a1
	mv	a0,s2
	mv	a1,s3
	call	__divdf3
	lw	a5,108(sp)
	lw	a2,216(a5)
	lw	a3,220(a5)
	call	__muldf3
	call	__truncdfsf2
	call	__extendsfdf2
	lw	a5,108(sp)
	mv	a2,a0
	mv	a3,a1
	addi	a0,a5,224
	call	printf
	lw	s3,224(sp)
	lw	s4,220(sp)
.L71:
	lw	ra,236(sp)
	lw	a0,120(sp)
	lw	s1,232(sp)
	lw	s2,228(sp)
	lw	s8,204(sp)
	lw	s10,196(sp)
	lw	s11,192(sp)
	addi	sp,sp,240
	jr	ra
.L227:
	li	a0,10
	call	putchar
	j	.L78
.L74:
 #APP
# 109 "../include/custom.h" 1
	rdcycle s4
# 0 "" 2
 #NO_APP
	lw	a4,104(sp)
 #APP
# 18 "../include/custom.h" 1
	addi zero,zero,0
.insn r 0x77, 0, 0, a5, a4, a5
# 0 "" 2
 #NO_APP
	j	.L81
.L226:
	lui	a0,%hi(.LANCHOR0)
	addi	a0,a0,%lo(.LANCHOR0)
	call	perror
	j	.L71
	.size	RVSCNN_conv_mp, .-RVSCNN_conv_mp
	.align	1
	.globl	error_check
	.type	error_check, @function
error_check:
	ble	a2,zero,.L243
	addi	sp,sp,-64
	sw	ra,60(sp)
	sw	s9,24(sp)
	ble	a3,zero,.L237
	mul	a5,a3,a3
	sw	s4,44(sp)
	lui	s4,%hi(.LANCHOR0+236)
	sw	s2,52(sp)
	sw	s3,48(sp)
	sw	s6,36(sp)
	sw	s7,32(sp)
	sw	s1,56(sp)
	sw	s5,40(sp)
	sw	s8,28(sp)
	sw	s10,20(sp)
	sw	s11,16(sp)
	mv	s2,a3
	mv	s7,a1
	mv	s6,a0
	sw	a2,12(sp)
	sw	a5,8(sp)
	addi	s4,s4,%lo(.LANCHOR0+236)
	sw	zero,4(sp)
	li	s3,0
	li	s9,0
.L234:
	lw	s5,4(sp)
	li	s1,0
.L236:
	slli	s11,s5,2
	add	s8,s6,s11
	li	s10,0
	add	s11,s7,s11
.L232:
	lw	a4,0(s8)
	lw	a5,0(s11)
	mv	a3,s10
	srai	a1,a4,31
	srai	a2,a5,31
	sub	a7,a4,a5
	sub	a1,a1,a2
	sgtu	a2,a7,a4
	sub	a1,a1,a2
	srai	a2,a1,31
	xor	a7,a2,a7
	sub	a6,a7,a2
	xor	a1,a2,a1
	sub	a1,a1,a2
	sgtu	a7,a6,a7
	sub	a7,a1,a7
	mv	a2,s1
	mv	a1,s3
	mv	a0,s4
	addi	s10,s10,1
	add	s9,s9,a6
	addi	s8,s8,4
	call	printf
	addi	s11,s11,4
	bne	s2,s10,.L232
	addi	s1,s1,1
	add	s5,s5,s2
	bne	s1,s2,.L236
	lw	a5,4(sp)
	lw	a4,8(sp)
	addi	s3,s3,1
	add	a5,a5,a4
	sw	a5,4(sp)
	lw	a5,12(sp)
	bne	a5,s3,.L234
	lw	s1,56(sp)
	lw	s2,52(sp)
	lw	s3,48(sp)
	lw	s4,44(sp)
	lw	s5,40(sp)
	lw	s6,36(sp)
	lw	s7,32(sp)
	lw	s8,28(sp)
	lw	s10,20(sp)
	lw	s11,16(sp)
.L235:
	lw	ra,60(sp)
	mv	a0,s9
	lw	s9,24(sp)
	addi	sp,sp,64
	jr	ra
.L243:
	li	a0,0
	ret
.L237:
	li	s9,0
	j	.L235
	.size	error_check, .-error_check
	.section	.text.startup,"ax",@progbits
	.align	1
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-80
	li	a0,16
	sw	s1,72(sp)
	sw	ra,76(sp)
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
	mv	s1,a0
	li	a5,1
	li	a3,26
.L245:
	add	a4,s1,a5
	sb	a5,-1(a4)
	addi	a5,a5,1
	bne	a5,a3,.L245
	li	a5,131072
	addi	a5,a5,-256
	li	a4,33488896
	addi	a4,a4,1
	sw	a5,24(sp)
	li	a0,36
	li	a5,-1
	sw	a4,20(sp)
	sb	a5,28(sp)
	call	malloc
	lbu	a5,28(sp)
	lw	a3,20(sp)
	lw	a4,24(sp)
	sb	a5,8(a0)
	li	a2,4
	li	a5,1
	mv	a1,a5
	mv	s6,a0
	sw	a3,0(a0)
	sw	a4,4(a0)
	mv	a3,s1
	mv	a4,a0
	li	a6,0
	mv	a0,a2
	call	Base_conv
	li	a2,2
	li	a1,4
	sw	a0,12(sp)
	call	MaxPool2x2
	li	a5,1
	li	a2,4
	mv	a1,a5
	mv	a4,s6
	mv	a3,s1
	li	a6,0
	mv	s9,a0
	mv	a0,a2
	call	Base_conv
	li	a5,1
	li	a2,4
	mv	a1,a5
	mv	a4,s6
	mv	a3,s1
	li	a6,0
	addi	s3,a0,16
	mv	a0,a2
	call	RVSCNN_conv_mp
	lui	s7,%hi(.LANCHOR0)
	li	a3,4
	addi	s7,s7,%lo(.LANCHOR0)
	mv	s10,a0
	mv	a1,a3
	addi	a0,s7,300
	li	a2,1
	lui	s5,%hi(.LANCHOR0+328)
	lui	s2,%hi(.LC3)
	call	printf
	addi	s5,s5,%lo(.LANCHOR0+328)
	addi	s2,s2,%lo(.LC3)
	li	s4,0
	li	s8,4
.L247:
	mv	a1,s4
	mv	a0,s5
	call	printf
	addi	s11,s3,-16
.L246:
	lw	a1,0(s11)
	mv	a0,s2
	addi	s11,s11,8
	call	printf
	lw	a1,-4(s11)
	mv	a0,s2
	call	printf
	li	a0,10
	call	putchar
	bne	s3,s11,.L246
	li	a0,10
	addi	s4,s4,1
	call	putchar
	addi	s3,s3,16
	bne	s4,s8,.L247
	addi	a0,s7,344
	call	puts
	mv	s4,s9
	li	s3,0
	li	s8,4
.L248:
	mv	a1,s3
	mv	a0,s5
	call	printf
	lw	a1,0(s4)
	mv	a0,s2
	addi	s3,s3,1
	call	printf
	li	a0,10
	call	putchar
	li	a0,10
	call	putchar
	addi	s4,s4,4
	bne	s3,s8,.L248
	addi	a0,s7,364
	call	puts
	mv	s4,s10
	li	s3,0
	li	s8,4
.L249:
	mv	a1,s3
	mv	a0,s5
	call	printf
	lw	a1,0(s4)
	mv	a0,s2
	addi	s3,s3,1
	call	printf
	li	a0,10
	call	putchar
	li	a0,10
	call	putchar
	addi	s4,s4,4
	bne	s3,s8,.L249
	mv	a2,s3
	li	a3,1
	mv	a1,s10
	mv	a0,s9
	call	error_check
	mv	a1,a0
	addi	a0,s7,400
	call	printf
	mv	a0,s1
	call	free
	mv	a0,s6
	call	free
	lw	a0,12(sp)
	call	free
	mv	a0,s9
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
	.string	"calloc poutputs"
.LC1:
	.string	"Filter values:"
	.zero	1
.LC2:
	.string	"Group %d:\n"
	.zero	1
.LC3:
	.string	"%4d "
	.zero	3
.LC4:
	.string	"Tile (m=%d, n=%d):\n"
.LC5:
	.string	"POOL_WB_INT channel 1: %d\n"
	.zero	1
.LC6:
	.string	"POOL_WB_INT channel 2: %d\n"
	.zero	1
.LC7:
	.string	"POOL_WB_INT channel 3: %d\n"
	.zero	1
.LC8:
	.string	"POOL_WB_INT channel 4: %d\n"
	.zero	1
.LC9:
	.string	"Accelerator cycle count: %d\n"
	.zero	3
.LC10:
	.word	858993459
	.word	1070805811
.LC11:
	.string	"GOPS: %.4f\n"
.LC12:
	.string	"Error check: ch=%d, i=%d, j=%d, cpu=%d, accel=%d, diff=%lld\n"
	.zero	3
.LC13:
	.string	"inside:%d c_in:%d c_out:%d\n"
.LC14:
	.string	"Channel %d:\n"
	.zero	3
.LC15:
	.string	"\nMaxPool2x2 output:"
.LC16:
	.string	"Accelerator RVSCNN_conv_mp output:"
	.zero	1
.LC17:
	.string	"Total error (CPU vs Accelerator): %d\n"
	.globl	__extendsfdf2
	.globl	__truncdfsf2
	.globl	__muldf3
	.globl	__divdf3
	.globl	__floatsidf
	.ident	"GCC: (g1b306039a) 15.1.0"
	.section	.note.GNU-stack,"",@progbits
