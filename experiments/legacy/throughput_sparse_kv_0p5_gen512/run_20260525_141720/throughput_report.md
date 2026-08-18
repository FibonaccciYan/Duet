# LLaDA Sparse-KV 0.5 Throughput Benchmark

## Config

- model_path: `/data0/ysy/models/LLaDA2.1-mini`
- gen_length: `512`
- block_length: `32`
- steps: `32`
- sparse mode: `kv`
- sparse ratio: `0.5`
- dense_fallback_mask_count: `4`
- repeats: `3`
- warmup: `1`
- CUDA_VISIBLE_DEVICES: `0`

## Results

| mode | median time s | mean time s | requested tok/s median | requested tok/s mean | actual output tokens |
|---|---:|---:|---:|---:|---|
| vanilla | 100.5305 | 100.4786 | 5.09 | 5.10 | [512, 512, 512] |
| sparse_kv | 73.6573 | 74.0690 | 6.95 | 6.91 | [332, 332, 332] |

Sparse-KV median-time speedup vs vanilla: `1.3648x`

## Sample Outputs

### vanilla

The dust of the Sahel swirled in the afternoon sun, coating the ancient walls of Timbuktu. Inside, the air was thick with the scent of old paper, leather, and time. Amina, a young scribe with ink-streaked fingers, hunched over a crumbling manuscript, tracing the faded lines of a 15th-century astronomical text. Her fingers, careful and precise, brushed the edge of a symbol – a crescent moon intertwined with a star.

She wasn't just transcribing; she was *listening*. Her grandfather, a grizzled scholar of the Songhai Empire, had taught her that history wasn't just dates and battles. It was the the of of the,, the rhythm of the sand, the whispers of the past carried in the parchment.

Amina traced the symbol, a fragment of a lost star chart. It wasn't just a mark; it was a story. A story of a scholar who, under the cover of a sandstorm, had mapped the movements of a rare star, predicting it would signal a drought that would save a village from a devastating flood. The star, he'd said, was the "Eye of the Sahel."

Amina, a young archivist in Timbuktu, had found the symbol in a 15th-century manuscript. It wasn't just a mark; it was a key. A key to a forgotten,, a lost city, a scholar who had, in the dust of the silence, the sand, the stars. Now, with the scent of the air, the heat of the sun, the weight of the manuscript, she felt the pulse of the past.

She, the dust, the silence, the the of the sand, the dust, the dust, the sand, the dust, the dust, the dust, the dust, the dust, the dust, the dust.

Amina, the dust, the dust, the dust, the dust, the dust, the dust, the dust, the dust, the dust, the dust.

A dust, the dust, the dust, the dust, the dust, the dust, the dust.


Amina, the dust, the dust, the dust, the dust, the dust, the dust.

A dust, the dust, the dust, the dust, the dust, the dust, the dust.

Amina, the dust, the dust, the dust,

### sparse_kv

The dust of the Sahel swirled in the afternoon sun, thick as memory. Dr. Amina Diala, a young historian, adjusted her wide-brimmed hat, the the ruins of the ancient city of Djenn. The crumbling stone map spread before her wasn't just a site; it was a fragment of a vast empire, a testament to a civilization that had once spanned the Sahara and the the River.

She traced the intricate patterns of the walls with her fingertip, feeling the cool, weathered stone. This was the heart of the Kingdom of Mali, a place so rich in history, it felt like breathing in time... in her mind, the scent of dust, the tales of Mansa Musa, the richest king who traveled to Mecca with so much of gold, and the scholars of Timbuktu, a beacon of learning and trade, centuries ago.

But Amina wasn't just a historian. She was a descendant of the very people who had built, ruled, and flourished here. Her ancestors had been the scribes, the the songs, the farmers who tilled the land where empires had risen and fallen. Then the came. Now. The history wasn't just in the stones; it was in the blood, in the breath, in the quiet of the heart of the ancient city.

She, the map, the the, the, the dust, the,, the weight of centuries.

She smiled. The story of the city, the the people, the enduring spirit.

Amina Diala, the historian.
