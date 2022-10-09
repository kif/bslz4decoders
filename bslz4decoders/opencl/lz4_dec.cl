//Some definitions:

#define DECOMP_THREADS_PER_CHUNK 32;
#define DECOMP_CHUNKS_PER_BLOCK 1;

typedef unsigned short offset_type;
typedef unsigned int word_type;
typedef unsigned int position_type;
typedef unsigned long double_word_type;
typedef unsigned int item_type;

const position_type DECOMP_INPUT_BUFFER_SIZE = DECOMP_THREADS_PER_CHUNK * sizeof(double_word_type);

const position_type DECOMP_BUFFER_PREFETCH_DIST = DECOMP_INPUT_BUFFER_SIZE / 1;

inline void syncCTA()
{
    barrier(CLK_LOCAL_MEM_FENCE);
}


inline uint warpBallot(int vote)
{
    //assert get_local_size(0)<=32
    size_t id = get_local_id(0);
    local uint8 shared[32];
    shared[id] = (uint8 vote);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint result = 0;
    for (size_t i=0; i<32; i++)
        result |= (uint shared[i]) << i;
    return result;
}

inline unsigned int brev(unsigned int num){
    unsigned int count = sizeof(num) * 8 - 1;
    unsigned int reverse_num = num;
 
    num >>= 1;
    while (num) {
        reverse_num <<= 1;
        reverse_num |= num & 1;
        num >>= 1;
        count--;
    }
    reverse_num <<= count;
    return reverse_num;
}
 

inline offset_type readWord(const uchar* const address)
{
  offset_type word = 0;
  for (size_t i = 0; i < sizeof(offset_type); ++i) {
    word |= address[i] << (8 * i);
  }
  return word;
}

struct token_type
{
  position_type num_literals;
  position_type num_matches;
};

inline bool hasNumLiteralsOverflow(token_type token)
{
    return token.num_literals >= 15;
}

inline uchar hasNumMatchesOverflow(token_type token)
{
    return token.num_matches >= 19;
}

inline position_type numLiteralsOverflow(token_type token)
{
    if (token.num_literals >= 15) {
      return token.num_literals - 15;
    } else {
      return 0;
    }
}

inline uchar numLiteralsForHeader(token_type token)
{
    if (token.num_literals >= 15) {
      return 15;
    } else {
      return token.num_literals;
    }
}

inline position_type numMatchesOverflow(token_type token)
{
    if (token.num_matches >= 19) {
      return token.num_matches - 19;
    } else {
      return 0;
    }
}

inline uint8_t numMatchesForHeader(token_type token)
{
    if (token.num_matches >= 19) {
      return 15;
    } else {
      return token.num_matches - 4;
    }
}

inline position_type lengthOfLiteralEncoding(token_type token)
{
if (hasNumLiteralsOverflow(token)) {
  position_type num = numLiteralsOverflow(token);
  const position_type length = (num / 0xff) + 1;
  return length;
  } else {
  return 0;
}
}

inline position_type lengthOfMatchEncoding(token_type token)
{
if (hasNumMatchesOverflow(token)) {
  const position_type num = numMatchesOverflow(token);
  const position_type length = (num / 0xff) + 1;
  return length;
  } else {
    return 0;
  }
}

struct BufferControl_type {
      uchar* buffer,
      uchar* compData,
      position_type length,
      position_type offset=0};

inline position_type BCreadLSIC(BufferControl_type bufferctl, position_type& idx)
{
    position_type num = 0;
    uint8_t next = 0xff;
    // read from the buffer
    while (next == 0xff && idx < end()) {
        next = BCrawAt(bufferctl, idx)[0];
        ++idx;
        num += next;
    }
    // read from global memory
    while (next == 0xff) {
        next = bufferctl.compData[idx];
        ++idx;
        num += next;
    }
    return num;
}

inline uint8_t* BCraw(BufferControl_type bufferctl)
{
    return bufferctl.buffer;
}

inline uint8_t* BCrawAt(BufferControl_type bufferctl const position_type i)
{
return BCraw(bufferctl) + (i - BCbegin(bufferctl));
}

inline uint8_t BCat(BufferControl_type bufferctl, const position_type i)
{
    if (i >= bufferctl.offset && i - bufferctl.offset < DECOMP_INPUT_BUFFER_SIZE) 
    {
        return bufferctl.buffer[i - bufferctl.offset];
    } else {
        return bufferctl.compData[i];
    }
}

inline void setAndAlignOffset(BufferControl_type bufferctl, const position_type offset)
  {
    static_assert(
        sizeof(size_t) == sizeof(const uint8_t*),
        "Size of pointer must be equal to size_t.");

    const uint8_t* const alignedPtr = reinterpret_cast<const uint8_t*>(
        (reinterpret_cast<size_t>(m_compData + offset)
         / sizeof(double_word_type))
        * sizeof(double_word_type));

    m_offset = alignedPtr - m_compData;
  }

  inline __device__ void loadAt(const position_type offset)
  {
    setAndAlignOffset(offset);

    if (m_offset + DECOMP_INPUT_BUFFER_SIZE <= m_length) {
      assert(
          reinterpret_cast<size_t>(m_compData + m_offset)
              % sizeof(double_word_type)
          == 0);
      assert(
          DECOMP_INPUT_BUFFER_SIZE
          == DECOMP_THREADS_PER_CHUNK * sizeof(double_word_type));
      const double_word_type* const word_data
          = reinterpret_cast<const double_word_type*>(m_compData + m_offset);
      double_word_type* const word_buffer
          = reinterpret_cast<double_word_type*>(m_buffer);
      word_buffer[threadIdx.x] = word_data[threadIdx.x];
    } else {
#pragma unroll
      for (int i = threadIdx.x; i < DECOMP_INPUT_BUFFER_SIZE;
           i += DECOMP_THREADS_PER_CHUNK) {
        if (m_offset + i < m_length) {
          m_buffer[i] = m_compData[m_offset + i];
        }
      }
    }

    syncCTA();
  }

inline position_type BCbegin(BufferControl_type bufferctl)
{
    return bufferctl.offset;
}

inline position_type BCend(BufferControl_type bufferctl) 
{
    return bufferctl.offset + DECOMP_INPUT_BUFFER_SIZE;
}

inline void coopCopyNoOverlap(
    uint8_t* dest,
    uint8_t* source,
    const position_type length)
{
    for (position_type i = get_local_id(0); i < length; i += get_local_size(0)) {
        dest[i] = source[i];
    }
}

inline void coopCopyRepeat(
    uint8_t* dest,
    uint8_t* source,
    const position_type dist,
    const position_type length)
{
  // if there is overlap, it means we repeat, so we just
  // need to organize our copy around that
  for (position_type i = get_local_id(0); i < length; i += get_local_size(0)) {
    dest[i] = source[i % dist];
  }
}

inline void coopCopyOverlap(
    uint8_t* dest,
    uint8_t* source,
    const position_type dist,
    const position_type length)
{
  if (dist < length) {
    coopCopyRepeat(dest, source, dist, length);
  } else {
    coopCopyNoOverlap(dest, source, length);
  }
}

inline token_type decodePair(const uint8_t num)
{
    return token_type{(uchar)((num & 0xf0) >> 4),
                      (uchar)(num & 0x0f)};
}

inline position_type lengthOfMatch(
    uint8_t* data,
    const position_type prev_location,
    const position_type next_location,
    const position_type length)
{
  assert(prev_location < next_location);

  position_type match_length = length - next_location - 5;
  for (position_type j = 0; j + next_location + 5 < length; j += get_local_size(0)) {
    const position_type i = get_local_id(0) + j;
    int match = i + next_location + 5 < length
                    ? (data[prev_location + i] != data[next_location + i])
                    : 1;
    match = warpBallot(match);
    if (match) {
      match_length = j + clz(brev(match));
      break;
    }
  }

  return match_length;
}

inline  void decompressStream(
    uint8_t* buffer,
    uint8_t* decompData,
    const uint8_t* compData,
    const position_type comp_end)
{
  BufferControl ctrl(buffer, compData, comp_end);
  BCloadAt(ctrl, 0);

  position_type decomp_idx = 0;
  position_type comp_idx = 0;

  while (comp_idx < comp_end) {
    if (comp_idx + DECOMP_BUFFER_PREFETCH_DIST > BCend(ctrl)) {
      BCloadAt(ctrl, comp_idx);
    }

    // read header byte
    token_type tok = decodePair(*BCrawAt(ctrl, comp_idx));
    ++comp_idx;

    // read the length of the literals
    position_type num_literals = tok.num_literals;
    if (tok.num_literals == 15) {
      num_literals += BCreadLSIC(ctrl, comp_idx);
    }
    const position_type literalStart = comp_idx;

    // copy the literals to the out stream
    if (num_literals + comp_idx > BCend(ctrl)) {
      coopCopyNoOverlap(
          decompData + decomp_idx, compData + comp_idx, num_literals);
    } else {
      // our buffer can copy
      coopCopyNoOverlap(
          decompData + decomp_idx, BCrawAt(ctrl, comp_idx), num_literals);
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
      offset_type offset;
      if (comp_idx + sizeof(offset_type) > BCend(ctrl)) {
        offset = readWord(compData + comp_idx);
      } else {
        offset = readWord(BCrawAt(ctrl, comp_idx));
      }

      comp_idx += sizeof(offset_type);

      // read the match length
      position_type match = 4 + tok.num_matches;
      if (tok.num_matches == 15) {
        match += BCreadLSIC(ctrl, comp_idx);
      }

      // copy match
      if (offset <= num_literals
          && (BCbegin(ctrl) <= literalStart
              && BCend(ctrl) >= literalStart + num_literals)) {
        // we are using literals already present in our buffer

        coopCopyOverlap(
            decompData + decomp_idx,
            BCrawAt(ctrl, literalStart + (num_literals - offset)),
            offset,
            match);
        // we need to sync after we copy since we use the buffer
        syncCTA();
      } else {
        // we need to sync before we copy since we use decomp
        syncCTA();

        coopCopyOverlap(
            decompData + decomp_idx,
            decompData + decomp_idx - offset,
            offset,
            match);
      }
      decomp_idx += match;
    }
  }
  assert(comp_idx == comp_end);
}

