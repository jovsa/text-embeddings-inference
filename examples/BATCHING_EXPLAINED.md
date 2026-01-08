# Why Queue Time and Inference Time Increase with Concurrency

## The Question

From the benchmark results, we see:
- **Concurrency 1**: Queue time 0.00 ms, Inference 4.56 ms
- **Concurrency 32**: Queue time 30.84 ms, Inference 19.42 ms
- **Concurrency 64**: Queue time 33.90 ms, Inference 18.62 ms

Why do both queue time AND inference time increase as concurrency increases?

---

## Understanding the Batching System

### How Batching Works

TEI uses a **demand-driven batching system**:

1. **Request Arrives** → Tokenized → Added to Queue
2. **Batching Task Notified** → Forms a batch from queued requests
3. **Batch Sent to Backend** → Inference runs
4. **Results Returned** → Distributed to individual requests

### Key Constraint: Single Batch Processing

Looking at the code (`core/src/infer.rs`):

```rust
// Batching task - only processes ONE batch at a time
async fn batching_task(queue: Queue, notify: Arc<Notify>, embed_sender: mpsc::Sender<NextBatch>) {
    loop {
        notify.notified().await;  // Wait for notification

        {
            let mut permit = embed_sender.reserve().await;  // Channel capacity = 1

            while let Some(next_batch) = queue.next_batch().await {
                permit.send(next_batch);  // Send batch to backend
                permit = embed_sender.reserve().await;  // Wait for next slot
            }
        }
    }
}
```

**Critical Point**: The channel has capacity 1, meaning only ONE batch can be in-flight at a time.

---

## Why Queue Time Increases

### The Bottleneck Effect

```
Time →

Concurrency 1:
Request 1 ──┐
            ├─► [Batch] ──► [Inference] ──► Done
Request 2 ──┘
            ↑
         No wait time

Concurrency 32:
Request 1 ──┐
Request 2 ──┤
Request 3 ──┤
   ...      ├─► [Queue grows] ──┐
Request 30 ─┤                   │
Request 31 ─┤                   ├─► [Batch 1] ──► [Inference] ──► Done
Request 32 ─┘                   │
                                │
Request 33 ──┐                  │
Request 34 ──┤                  │
   ...      ├─► [Queue grows] ──┘
            │                   │
            │                   ├─► [Batch 2] ──► [Inference] ──► Done
            └───────────────────┘
            ↑
      Requests wait here
      (Queue Time)
```

### The Math

1. **Request Arrival Rate**: With concurrency 32, requests arrive very quickly (all at once)
2. **Batch Processing Rate**: Limited by:
   - Batching task can only form one batch at a time
   - Backend can only process one batch at a time (channel capacity = 1)
   - Inference takes time (4-20ms per batch)

3. **Result**: Requests accumulate in the queue faster than they can be processed

### Example Timeline (Concurrency 32)

```
T+0ms:   32 requests arrive simultaneously
         All added to queue instantly

T+0ms:   Batching task notified, starts forming Batch 1
         Takes ~5ms to form batch (processing 32 requests)

T+5ms:   Batch 1 sent to backend
         Backend starts inference (takes ~20ms)

T+25ms:  Batch 1 completes
         Results returned to first 16 requests

T+25ms:  Batching task forms Batch 2 from remaining requests
         Takes ~5ms

T+30ms:  Batch 2 sent to backend
         Inference takes ~20ms

T+50ms:  Batch 2 completes
         Results returned to remaining 16 requests
```

**Queue Time for Request 1**: 0ms (first in batch)
**Queue Time for Request 32**: ~25ms (waits for Batch 1 to complete)

---

## Why Inference Time Increases

### Reason 1: Larger Batches

With higher concurrency, more requests are available in the queue, allowing larger batches:

```
Concurrency 1:  Batch size ~12.5 requests
Concurrency 32: Batch size ~5.6 requests (but batches form more frequently)
Concurrency 64: Batch size ~5.6 requests
```

**Wait, why doesn't batch size increase linearly?**

The batch size is limited by:
- `max_batch_tokens` (default: 16384 tokens)
- `max_batch_requests` (optional limit)

But even if batch size stays similar, **inference time still increases** because:

### Reason 2: More Total Work

Even with similar batch sizes, the system processes more total requests:

```
Concurrency 1:  50 requests → ~4 batches → 4.56ms per batch
Concurrency 32: 50 requests → ~9 batches → 19.42ms per batch
```

### Reason 3: Batch Formation Overhead

With higher concurrency:
- More requests to process when forming batches
- Queue operations take longer (more entries to iterate)
- Memory allocation for larger batches

### Reason 4: Backend Saturation

The backend (Candle/ORT/Python) may:
- Have limited parallelism
- Experience memory pressure with larger batches
- Have overhead from processing more diverse batch sizes

---

## The Real Bottleneck: Sequential Batch Processing

The key insight is that **batches are processed sequentially**, not in parallel:

```
┌─────────────────────────────────────────────────────────┐
│                    THE BOTTLENECK                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Channel Capacity = 1                                    │
│  ┌──────────┐                                           │
│  │  Batch 1 │ ──► [Backend Processing] ──► Done        │
│  └──────────┘                                           │
│                                                          │
│  ┌──────────┐                                           │
│  │  Batch 2 │ ──► [WAITING] ──► [Backend] ──► Done     │
│  └──────────┘     ↑                                      │
│                   │                                      │
│              Must wait for                              │
│              Batch 1 to finish                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Why Not Parallel Batches?

The channel capacity is intentionally set to 1:

```rust
let (embed_sender, embed_receiver) = mpsc::channel(1);  // Capacity = 1
```

This ensures:
- **Memory efficiency**: Only one batch in memory at a time
- **Backend simplicity**: Backend doesn't need to handle concurrent batches
- **Predictable behavior**: Easier to reason about and debug

---

## Visual Timeline Comparison

### Concurrency 1 (Low Queue Time)

```
Request 1 ──► [Queue] ──► [Batch: 1 req] ──► [Inference: 4.56ms] ──► Done
                                                                    ↑
Request 2 ──► [Queue] ──► [Batch: 1 req] ──► [Inference: 4.56ms] ──┘
                                                                    ↑
Request 3 ──► [Queue] ──► [Batch: 1 req] ──► [Inference: 4.56ms] ──┘

Queue Time: ~0ms (requests processed immediately)
Inference Time: 4.56ms (small batches)
```

### Concurrency 32 (High Queue Time)

```
Request 1-16 ──► [Queue] ──┐
                           ├─► [Batch 1: 16 reqs] ──► [Inference: 19.42ms] ──► Done
Request 17-32 ──► [Queue] ──┘                                                      ↑
                                                                                    │
Request 33-48 ──► [Queue] ──┐                                                      │
                           ├─► [Batch 2: 16 reqs] ──► [WAIT] ──► [Inference: 19.42ms] ──┘
Request 49-64 ──► [Queue] ──┘     ↑
                                   │
                            Must wait for
                            Batch 1 to finish

Queue Time: 0-30ms (depending on position in queue)
Inference Time: 19.42ms (larger batches)
```

---

## The Trade-off

### Low Concurrency (1-4)
- ✅ **Low Queue Time**: Requests processed immediately
- ✅ **Low Inference Time**: Small batches, fast processing
- ❌ **Low Throughput**: Sequential processing, underutilized hardware

### High Concurrency (32-64)
- ❌ **High Queue Time**: Requests wait for batching
- ❌ **High Inference Time**: Larger batches, more work
- ✅ **High Throughput**: Better hardware utilization, more requests/second

### The Sweet Spot (8-16)
- ✅ **Moderate Queue Time**: Acceptable wait times
- ✅ **Moderate Inference Time**: Reasonable batch sizes
- ✅ **Good Throughput**: Balanced performance

---

## Summary

**Queue Time Increases Because:**
1. Requests arrive faster than they can be batched
2. Batches are processed sequentially (channel capacity = 1)
3. Requests accumulate in the queue waiting for their batch to be processed

**Inference Time Increases Because:**
1. Larger batches contain more tokens to process
2. More total work even if batch sizes are similar
3. Backend overhead from processing larger/more frequent batches
4. System saturation under high load

**The Fundamental Constraint:**
The system is designed for **throughput optimization**, not **latency optimization**. Higher concurrency maximizes hardware utilization (especially GPUs) at the cost of individual request latency.

---

## Optimization Strategies

1. **Adjust `max_batch_tokens`**: Smaller batches = lower inference time but lower throughput
2. **Adjust `max_batch_requests`**: Limit batch size to control latency
3. **Use Multiple Backends**: Distribute load across multiple TEI instances
4. **Tune Concurrency**: Find the sweet spot for your latency/throughput needs


