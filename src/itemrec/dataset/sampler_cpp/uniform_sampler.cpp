// ----------------------------------------------------------------------------
// ItemRec / Item Recommendation Benchmark
// Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
// ----------------------------------------------------------------------------
// C++ Sampling Implementation: Uniform Sampler
// ----------------------------------------------------------------------------

#include <torch/extension.h>
#include <vector>
#include <unordered_set>
#include <random>

// Helper Functions -----------------------------------------------------------
// set up the random seed
void set_seed(int64_t seed)
{
    srand(seed);
}
// generate a random integer in [a, b]
int64_t rand_int(int a, int b)
{
    return rand() % (b - a + 1) + a;
}

// Uniform Sampler ------------------------------------------------------------
// Args:
//  item_size: int64_t
//      The total number of items.
//  pos_items: std::vector<std::unordered_set<int64_t>>
//      The positive item set for each user.
//  N: int64_t
//      The number of negative samples for each user.
//  seed: int64_t
//      The random seed.
// Returns:
//  torch::Tensor (B, N)
//      The negative item IDs for each user.
// ----------------------------------------------------------------------------
torch::Tensor uniform_sample(
    int64_t item_size,
    std::vector<std::unordered_set<int64_t>> &pos_items,
    int64_t N,
    int64_t seed = -1
)
{
    int64_t B = pos_items.size();
    torch::Tensor neg_items = torch::empty({B, N}, torch::kInt64);
    int64_t *neg_items_ptr = neg_items.data_ptr<int64_t>();
    std::mt19937_64 gen(seed == -1 ? std::random_device{}() : seed);
    for (int64_t i = 0, offset = 0; i < B; i++) 
    {
        offset = i * N;
        std::unordered_set<int64_t> &pos_set = pos_items[i];
        for (int64_t count = 0; count < N; )
        {
            int64_t sampled_item = rand_int(0, item_size - 1);
            if (pos_set.find(sampled_item) == pos_set.end()) 
            {
                neg_items_ptr[offset + count] = sampled_item;
                count++;
            }
        }
    }
    return neg_items;
}

// Pybind11 Module ------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("uniform_sample", &uniform_sample, "Uniform Sampler (C++ Implementation)");
}
