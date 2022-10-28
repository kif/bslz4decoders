// Bitshuffle LZ4 decompressor

#ifndef LZ4_BLOCK_SIZE
# define LZ4_BLOCK_SIZE 8192
#endif
#define LZ4_BLOCK_EXTRA 400
#ifdef __ENDIAN_LITTLE__
#define SWAP_BE 1
#define SWAP_LE 0
#else
#define SWAP_BE 0
#define SWAP_LE 1
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
    return token.s1 >= 15;
}

//parse overflow, return the number of overflow and the new position
inline uint2 read_overflow(local uint8_t* buffer, 
                           position_t buffer_size,
                           position_t idx){
    position_t num = 0;
    uint8_t next = 0xff;
    while (next == 0xff && idx < buffer_size){
        next = buffer[idx];
        idx += 1;
        num += next;
    }
    return (uint2)(num, idx);
}

inline void copy_no_overlap(local uint8_t* dest,
                            const position_t dest_position,
                            local uint8_t* source,
                            const position_t src_position,
                            const position_t length){
    for (position_t i=get_local_id(0); i<length; i+=get_local_size(0)) {
        dest[dest_position+i] = source[src_position+i];
    }
}

inline void copy_repeat(local uint8_t* dest,
                        const position_t dest_position,
                        local uint8_t* source,
                        const position_t src_position,
                        const position_t dist,
                        const position_t length){
    
    // if there is overlap, it means we repeat, so we just
    // need to organize our copy around that
    for (position_t i=get_local_id(0); i<length; i+=get_local_size(0)) {
        dest[dest_position+i] = source[src_position + i%dist];
    }
}

inline void copy_collab(local uint8_t* dest,
                        const position_t dest_position,
                        local uint8_t* source,
                        const position_t src_position,    
                        const position_t dist,
                        const position_t length){
    //Generic copy function
    if (dist < length) {
        copy_repeat(dest, dest_position, source, src_position, dist, length);
    } 
    else {
        copy_no_overlap(dest, dest_position, source, src_position, length);
    }
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

uint16_t load16_at(local uint8_t *src,
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

//Decompress one block in shared memory
inline void lz4_decompress_local_block( local uint8_t* local_cmp,
                                        local uint8_t* local_dec,
                                        const uint32_t cmp_buffer_size,
                                        const uint32_t dec_buffer_size){
    
    uint32_t gid = get_group_id(0); // One block is decompressed by one workgroup
    uint32_t lid = get_local_id(0); // This is the thread position in the group...
    uint32_t wg = get_local_size(0); // workgroup size
        
    position_t dec_idx = 0;
    position_t cmp_idx = 0;
    while (cmp_idx < cmp_buffer_size) {
         // read header byte
        token_t tok = decode_token(local_cmp[cmp_idx]);
        // if (lid==0)        printf("gid %u at idx %u/%u. Token is litterials: %u; matches: %u\n", gid, cmp_idx, cmp_buffer_size,tok.s0, tok.s1);

        cmp_idx+=1;
        
        // read the length of the literals
        position_t num_literals = tok.s0;
        if (has_liter_over(tok)) {
            uint2 tmp = read_overflow(local_cmp, 
                                      cmp_buffer_size,
                                      cmp_idx);
            num_literals += tmp.s0;
            cmp_idx = tmp.s1;
        }
        const position_t start_literal = cmp_idx;

        // copy the literals to the dst stream in parallel
        // if (lid==0) printf("gid %u: copy literals from %u to %u <%u (len %u)\n", gid, cmp_idx,num_literals+cmp_idx,cmp_buffer_size,num_literals);
        copy_no_overlap(local_dec, dec_idx, local_cmp, cmp_idx, num_literals);
        cmp_idx += num_literals;
        dec_idx += num_literals;
        
        // Note that the last sequence stops right after literals field.
        // There are specific parsing rules to respect to be compatible with the
        // reference decoder : 1) The last 5 bytes are always literals 2) The last
        // match cannot start within the last 12 bytes Consequently, a file with
        // less then 13 bytes can only be represented as literals These rules are in
        // place to benefit speed and ensure buffer limits are never crossed.
        if (cmp_idx < cmp_buffer_size) {
            
          // read the offset
          uint16_t offset = load16_at(local_cmp, cmp_idx, SWAP_LE);
          // if (lid==0) printf("gid %u: offset is %u at %u\n",gid, offset, cmp_idx);
          if (offset == 0) {
              //corruped block
              if (lid == 0) 
                  printf("Corrupted block #%u\n", gid);
              return;
          }
          
          cmp_idx += 2;

          // read the match length
          position_t match = 4 + tok.s1;
          if (has_match_over(tok)) {
            uint2 tmp = read_overflow(local_cmp, 
                                      cmp_buffer_size,
                                      cmp_idx);
            match += tmp.s0;
            cmp_idx = tmp.s1;
          }

          //syncronize threads before reading shared memory
          barrier(CLK_LOCAL_MEM_FENCE);
          
          // copy match
          copy_collab(local_dec, dec_idx, local_dec, dec_idx - offset, offset, match);
          dec_idx += match;
        }
    }
    //syncronize threads before reading shared memory
    barrier(CLK_LOCAL_MEM_FENCE);
}

//Perform the bifshuffling
inline void bitunshuffle32( local uint8_t* inp,
                            local uint8_t* out,
                            const uint32_t buffer_size){ //8k
    uint32_t gid = get_group_id(0);
    uint32_t lid = get_local_id(0);
    uint32_t wg = get_local_size(0);
    uint32_t u32_buffer_size = buffer_size/4; //2k
    uint32_t offset = u32_buffer_size/32; //64
    // One thread deals with one or several output data
    for (uint32_t dpos=lid; dpos<u32_buffer_size; dpos+=wg){
        uint32_t res = 0;
        uint32_t u32_word_offset = dpos/32;
        uint32_t u32_bit_offset = dpos%32;
        // read bits at several places...
        for (uint32_t bit=0; bit<32; bit++){
            uint32_t u8_word_pos = 4*(bit*offset + u32_word_offset) + (u32_bit_offset/8);
            uint32_t u8_bit_pos = u32_bit_offset%8;
            // if ((dpos==0)&&(gid==0)) printf("gid %u: pixel #%u:%u read from (%u:%u)32bits (%u:%u)8bits \n",gid, dpos, bit, u32_word_offset+bit*offset,u32_bit_offset, u8_word_pos,u8_bit_pos);
            res |= ((inp[u8_word_pos]>>(u8_bit_pos)) & 1)<<bit;
        }
        uchar4 tmp = as_uchar4(res);
        out[4*dpos] = tmp.s0;
        out[4*dpos+1] = tmp.s1;
        out[4*dpos+2] = tmp.s2;
        out[4*dpos+3] = tmp.s3;
    }
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


kernel void bslz4_decompress_block(
    global uint8_t* comp_src,
    global uint8_t* dec_dest,
    global uint64_t* block_start,
    const uint8_t item_size){
    
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
    for (uint32_t i=lid; i<cmp_buffer_size; i+=wg){
        uint64_t read_pos = start_read + i;
        if (read_pos<end_read)
            local_cmp[i] = comp_src[read_pos];
        else
            local_cmp[i] = 0;            
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
    
    //All the work is performed here:
    lz4_decompress_local_block( local_cmp, local_dec, cmp_buffer_size, LZ4_BLOCK_SIZE);
    
    //Bitshuffle?
    barrier(CLK_LOCAL_MEM_FENCE);
    local uint8_t* local_buffer;
    //if item_size is 4, perform bit-unshuffle 
    if (item_size == 4){
        //32 bits data
        bitunshuffle32(local_dec, local_cmp, LZ4_BLOCK_SIZE);
        local_buffer=local_cmp;
    }
    else {
        local_buffer = local_dec;
    }
        
    //Finally copy the destination data from local to global memory:
    uint64_t start_write = LZ4_BLOCK_SIZE*gid;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint32_t i=lid; i<LZ4_BLOCK_SIZE; i+=wg){
        dec_dest[start_write + i] = local_buffer[i];
    }

}
