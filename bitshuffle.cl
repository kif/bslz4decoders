// Bitshuffle LZ4 decompressor

#ifndef LZ4_BLOCK_SIZE
# define LZ4_BLOCK_SIZE 8192
#endif
#define LZ4_BLOCK_EXTRA 400
#ifdef __ENDIAN_LITTLE__
#define SWAP_BE 1
#else
# define SWAP_BE 0
#endif


#define int8_t char
#define uint8_t uchar
#define int16_t short
#define uint16_t ushort
#define int32_t int
#define uint32_t uint
#define int64_t long
#define uint64_t ulong

#define position_t uint
#define token_t uchar2

/* Decompress and bitunshuffle a single block into local memory. 

Param: src: input buffer in global memory
       dst: output buffer in local memory, must be of size * elem_size bytes
     size : number of elements in input
elem_size : element size of typed data
block_size : Process in blocks of this many elements. Pass 0 to select automatically (recommended).

Returns: number of bytes consumed in *input* buffer, negative error-code if failed.

Hint on workgroup size: 64 threads, should handle bs8, bs16, bs32 and bs64.
*/
long bshuf_decompress_lz4_block(global uint8_t *src,
                                local uint8_t *dst,
                                global uint64_t *block_start,
                                const uint64_t size, //
                                const uint64_t elem_size) {
return 0;
}


inline token_t decode_token(uint8_t value){
    return (token_t)(value >> 4,     // literals
                     value & 0x0f);  // matches
}

inline bool has_liter_over(token_t token)
{
    return token.s0 >= 15;
}

inline bool has_match_over(token_t token)
{
    return token.s1 >= 19;
}

inline position_t num_liter_over(token_t token)
{
    return max((int)token.s0 - 15, 0); 
}

inline position_t num_match_over(token_t token)
{
    return max((int)token.s1 - 19, 0);
}

//parse overflow, return the number of overflow and the new position
inline uint2 read_overflow(local uint8_t* buffer, 
                           position_t buffer_size,
                           position_t idx){
    position_type num = 0;
    uint8_t next = 0xff;
    while (next == 0xff && idx < buffer_size){
        next = buffer[idx];
        idx += 1;
        num += next;
    }
    return (uint2)(num, idx);
}



uint64_t load64_at(global uint8_t *src,
                   const uint64_t position,
                   const bool swap){
    uchar8 vector;
    if (swap){
        vector = (uchar8)(src[position+7],src[position+6],
                          src[position+5],src[position+4],
                          src[position+3],src[position+2],
                          src[position+1],src[position+0]);
    }
    else{
        vector = (uchar8)(src[position+0],src[position+1],
                          src[position+2],src[position+3],
                          src[position+4],src[position+5],
                          src[position+6],src[position+7]);
    }
    return as_ulong(vector);
}

uint32_t load32_at(global uint8_t *src,
                   const uint64_t position,
                   const bool swap){
    uchar4 vector;
    if (swap){
        vector = (uchar4)(
                  src[position+3],src[position+2],
                  src[position+1],src[position+0]);
    }
    else{
        vector = (uchar4)(src[position+0],src[position+1],
                          src[position+2],src[position+3]);
    }
    return as_uint(vector);
}

uint16_t load16_at(global uint8_t *src,
                   const uint64_t position,
                   const bool swap){
    uchar2 vector;
    if (swap){
        vector = (uchar2)(src[position+1],src[position+0]);
    }
    else{
        vector = (uchar2)(src[position+0],src[position+1]);
    }
    return as_ushort(vector);
}


/* Preprocessing kernel which performs:
- Memset arrays
- read block position stored in block_position array

Param: 
- src: input buffer in global memory
- size: input buffer size
- block_position: output buffer in local memory containing the index of the begining of each block
- max_blocks: allocated memory for block_position array
- nb_blocks: output buffer with the actual number of blocks in src.

Return: Nothing, this is a kernel

Hint on workgroup size: little kernel ...
*/

kernel void lz4_unblock(global uint8_t *src,
                        const uint64_t size,
                        global uint64_t *block_start,
                        const uint32_t max_blocks,
                        global uint32_t *nb_blocks){
    // printf("\n SWAP %d \n",SWAP_BE);
    uint64_t total_nbytes = load64_at(src,0,SWAP_BE);
    uint32_t block_nbytes = load32_at(src,8,SWAP_BE);

    uint32_t block_idx = 0;
    uint64_t pos = 12;
    uint32_t block_size;
        
    while ((pos+4<size) && (block_idx<max_blocks)){
        block_size = load32_at(src, pos, SWAP_BE);
        block_start[block_idx] = pos + 4;
        block_idx +=1;
        pos += 4 + block_size;
    }
    nb_blocks[0] = block_idx;
}                    


kernel void lz4_decompress_block(
    global uint8_t* comp_src,
    global uint8_t* dec_dest,
    global uint64_t* block_start,
    const uint64_t comp_end)
{
    uint32_t gid = get_group_id(0); // One block is decompressed by one workgroup
    uint32_t lid = get_local_id(0); // This is the thread position in the group...
    uint32_t wg = get_local_size(0); // workgroup size
    
    // No need to guard, the number of blocks can be calculated in advance.
    uint64_t start_read = block_start[gid];
    if (start_read<12) return;
    
    local uint8_t local_cmp[LZ4_BLOCK_SIZE+LZ4_BLOCK_EXTRA];
    local uint8_t local_dec[LZ4_BLOCK_SIZE];
    
    uint32_t cmp_buffer_size = load32_at(comp_src, start_read-4, SWAP_BE); 
    uint64_t end_read = start_read + cmp_buffer_size;
    // Copy locally the compressed buffer and memset the destination buffer
    for (uint32_t i=0; i<cmp_buffer_size; i+=get_local_size(0)){
        uint64_t read_pos = start_read + i + lid;
        if (read_pos<end_read)
            local_cmp[i+lid] = comp_src[read_pos];
        else
            local_cmp[i+lid] = 0;            
    }
    for (uint32_t i=cmp_buffer_size; i<LZ4_BLOCK_SIZE+LZ4_BLOCK_EXTRA; i+=get_local_size(0)){
        uint64_t pos = i + lid;
        if (pos<LZ4_BLOCK_SIZE+LZ4_BLOCK_EXTRA)
            local_cmp[pos] = 0;
    }
    for (uint32_t i=0; i<LZ4_BLOCK_SIZE; i+=get_local_size(0)){
        uint64_t pos = i + lid;
        if (pos<LZ4_BLOCK_SIZE)
            local_dec[pos] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    position_t dec_idx = 0;
    position_t cmp_idx = 0;
    while (comp_idx < comp_end) {
        
         // read header byte
        token_t tok = decode_token(local_cmp[cmp_idx]);
        cmp_idx+=1;
        
        // read the length of the literals
        position_t num_literals = tok.num_literals;
        if (has_liter_over(tok)) {
            uint2 tmp = read_overflow(local_cmp, 
                                      cmp_buffer_size,
                                      comp_idx);
            num_literals += tmp.s0;
            cmp_idx = tmp.s1;
        }
        const position_t start_literal = comp_idx;

        // copy the literals to the dst stream
        for (position_t i=0; i<num_literals; i+=wg) {
            if (i<num_literals)
                local_dec[dec_idx+i] = local_cmp[cmp_idx+i];
        }
        comp_idx += num_literals;
        decomp_idx += num_literals;
        
        // Note that the last sequence stops right after literals field.
        // There are specific parsing rules to respect to be compatible with the
        // reference decoder : 1) The last 5 bytes are always literals 2) The last
        // match cannot start within the last 12 bytes Consequently, a file with
        // less then 13 bytes can only be represented as literals These rules are in
        // place to benefit speed and ensure buffer limits are never crossed.
        if (comp_idx < comp_end) {

          // read the offset
          uint16_t offset = load16_at(local_cmp, comp_idx, SWAP_BE);
          comp_idx += 2;

            #TODO
          // read the match length
          position_type match = 4 + tok.num_matches;
          if (tok.num_matches == 15) {
            match += BCreadLSIC(ctrl, comp_idx);
          }

}
