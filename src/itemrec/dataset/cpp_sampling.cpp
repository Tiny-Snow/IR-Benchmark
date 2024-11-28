// cppimport

// ----------------------------------------------------------------------------
// ItemRec / Item Recommendation Benchmark
// Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
// ----------------------------------------------------------------------------
// C++ Sampling Implementation
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cassert>
#include <random>
#include <cstdlib>

namespace py = pybind11;
using namespace pybind11::literals;

// Helper Functions -----------------------------------------------------------
// set up the random seed
void set_seed(int seed)
{
    srand(seed);
}
// generate a random integer in [a, b]
int rand_int(int a, int b)
{
    return rand() % (b - a + 1) + a;
}

// C++ Sampling ---------------------------------------------------------------
/*
def _py_sampling(item_num: int, sample_num: int, exclude_items: List[int]) -> List[int]:
    """
    ## Function
    Python implementation of sampling.
    This function samples `sample_num` items from `items` while excluding
    the items in `exclude_items`.

    ## Arguments
    item_num: int
        The number of items. The set of items is [0, item_num).
    sample_num: int
        The number of items to sample.
    exclude_items: List[int]
        The set of items to exclude.
    """
    remain_items = set(range(item_num)) - set(exclude_items)
    assert len(remain_items) >= sample_num, "When sampling, the sample_num should be less than items size."
    return random.sample(list(remain_items), sample_num)
*/
std::vector<int> cpp_sampling(int item_num, int sample_num, std::vector<int> exclude_items)
{
    std::set<int> sample_items;
    while ((int)sample_items.size() < sample_num)
    {
        int item = rand_int(0, item_num - 1);
        if (std::binary_search(exclude_items.begin(), exclude_items.end(), item))
            continue;   // the exclude_items is assumed to be sorted
        if (sample_items.find(item) != sample_items.end())
            continue;
        sample_items.insert(item);
    }
    return std::vector<int>(sample_items.begin(), sample_items.end());
}

// deprecated version
std::vector<int> cpp_sampling2(int item_num, int sample_num, std::vector<int> exclude_items)
{
    std::vector<int> remain_items;
    int m = exclude_items.size();
    int next_del = 0;
    // the exclude_items is assumed to be sorted
    for (int i = 0; i < item_num; i++)
    {
        if (next_del < m && i == exclude_items[next_del])
            next_del++;
        else
            remain_items.push_back(i);
    }
    assert(remain_items.size() >= sample_num);
    std::random_shuffle(remain_items.begin(), remain_items.end());
    std::vector<int> sample_items(remain_items.begin(), remain_items.begin() + sample_num);
    return sample_items;
}

// Python Binding -------------------------------------------------------------
PYBIND11_MODULE(cpp_sampling, m) 
{
    m.doc() = "Sampling C++ implementation";
    // set_seed: Set the random seed for sampling.
    m.def("set_seed", &set_seed, "Set the random seed for sampling.",
        "seed"_a);
    // sample: Sample `sample_num` items from `items` while excluding the items in `exclude_items`.
    // def _py_sampling(item_num: int, sample_num: int, exclude_items: List[int]) -> List[int]:
    m.def("sample", &cpp_sampling, "Sample `sample_num` items from `items` while excluding the items in `exclude_items`.",
        "item_num"_a, "sample_num"_a, "exclude_items"_a);
}

/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11', '-undefined dynamic_lookup']
%>
*/