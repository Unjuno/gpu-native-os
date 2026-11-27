# GPU-native guest OS v0.1. All waves start at boot.
# Scalar registers s*, vector registers v*. Device inputs: wid, lanes, roots.
# Sequentially consistent shared word memory. Only the guest changes its RAM.
.const BOOT 0
.const HEAD 1
.const TAIL 2
.const LIVE 3
.const QBASE 16

boot:
    # Derive memory layout independently on every wave.
    SET s_executed 0
    MUL s_cap roots 2
    MUL s_scratch s_cap 3
    ADD s_scratch s_scratch QBASE
    ADD s_output s_scratch roots
    ADD s_count1 s_output roots
    ADD s_count2 s_count1 roots
    BNE wid 0 wait_boot
    STORE HEAD 0
    STORE TAIL 0
    STORE LIVE roots
    SET s_i 0
boot_enqueue:
    BEQ s_i roots boot_publish
    MUL s_slot s_i 3
    ADD s_slot s_slot QBASE
    STORE s_slot 1
    ADD s_addr s_slot 1
    STORE s_addr s_i
    ADD s_addr s_slot 2
    STORE s_addr 1
    ADD s_i s_i 1
    JMP boot_enqueue
boot_publish:
    STORE TAIL roots
    STORE BOOT 1
wait_boot:
    LOAD s_flag BOOT
    BEQ s_flag 0 wait_boot

# Decentralized scheduler. Every wave executes this same code.
# CAS prevents two waves from claiming the same ticket.
schedule:
    LOAD s_head HEAD
    LOAD s_tail TAIL
    BLT s_head s_tail claim
    LOAD s_live LIVE
    BEQ s_live 0 halt
    JMP schedule
claim:
    ADD s_next s_head 1
    CAS s_old HEAD s_head s_next
    BNE s_old s_head schedule
    MUL s_slot s_head 3
    ADD s_slot s_slot QBASE
    ADD s_addr s_slot 2
wait_slot:
    LOAD s_flag s_addr
    BEQ s_flag 0 wait_slot
    LOAD s_kind s_slot
    ADD s_addr s_slot 1
    LOAD s_id s_addr
    BLT s_id 0 fault
    BLT s_id roots dispatch
    JMP fault
dispatch:
    BEQ s_kind 1 stage1
    BEQ s_kind 2 stage2
    JMP fault

# Root job: each lane squares its own element; wave reduces the vector.
# The guest spawns a dependent child job, without a host call.
stage1:
    MUL s_base s_id lanes
    ADD s_base s_base 1
    VID v_x
    VADD_S v_x v_x s_base
    VMUL v_sq v_x v_x
    VREDUCE_SUM s_sum v_sq
    ADD s_addr s_scratch s_id
    STORE s_addr s_sum
    ADD s_addr s_count1 s_id
    ATOM_ADD s_old s_addr 1
    BNE s_old 0 fault

    # Account for child BEFORE publishing it, so LIVE cannot reach zero early.
    ATOM_ADD s_old LIVE 1
    ATOM_ADD s_ticket TAIL 1
    BLT s_ticket s_cap publish_child
    JMP fault
publish_child:
    MUL s_slot s_ticket 3
    ADD s_slot s_slot QBASE
    STORE s_slot 2
    ADD s_addr s_slot 1
    STORE s_addr s_id
    ADD s_addr s_slot 2
    STORE s_addr 1
    JMP complete

# Child job: consumes its parent's result, then writes a final result.
stage2:
    ADD s_addr s_scratch s_id
    LOAD s_sum s_addr
    MUL s_result s_sum 3
    ADD s_result s_result s_id
    ADD s_addr s_output s_id
    STORE s_addr s_result
    ADD s_addr s_count2 s_id
    ATOM_ADD s_old s_addr 1
    BNE s_old 0 fault
complete:
    ADD s_executed s_executed 1
    ATOM_ADD s_old LIVE -1
    BLT 0 s_old schedule
    JMP fault
halt:
    HALT
fault:
    TRAP
