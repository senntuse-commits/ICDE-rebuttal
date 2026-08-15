#include "../include/DupAggtree.h"
#include "../include/PrimitiveProfile.h"
#include <climits>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace
{
size_t checkedElementCount(size_t rows, size_t cols)
{
    const size_t maxElements = vector<int>().max_size();
    if (cols != 0 && rows > maxElements / cols)
        throw length_error("DupAggtree element count exceeds vector capacity");
    return rows * cols;
}
}

DupAggtree::DupAggtree(const vector<vector<int>> &D, const vector<int> &B)
{
    if (D.empty())
#ifdef SGX_ENCLAVE_BUILD
        return;
#else
        throw invalid_argument("Data matrix D is empty.");
#endif
    this->D = D;
    this->B = B;
}

DupAggtree::DupAggtree(vector<vector<int>> &&D, vector<int> &&B)
    : D(std::move(D)), B(std::move(B))
{
    if (this->D.empty())
#ifdef SGX_ENCLAVE_BUILD
        return;
#else
        throw invalid_argument("Data matrix D is empty.");
#endif
}

DupAggtree::DupAggtree(const vector<int>& D_flat, int n_rows, int n_cols, const vector<int>& B_flat)
{
    this->D_flat      = &D_flat;
    this->n_rows_flat = n_rows;
    this->n_cols_flat = n_cols;
    this->B_flat      = &B_flat;
}

// Flat-array implementation: replaces the original per-node vector<int> px/lpx
// with a single pre-allocated block, eliminating O(n log n) small heap allocations.
//
// Layout: node (h, i) stores px at
// A_px[(loff[h]+i)*dim .. +dim-1].  A parent's former lpx value is exactly
// its left child's upstream px.  The downstream pass reads that child before
// overwriting it, so a second full-size lpx array is unnecessary.
vector<vector<int>> DupAggtree::DupAggtreeRun()
{
    PrimitiveProfileScope primitiveScope(PrimitiveAggtree);
    if (D.empty() || D[0].empty())
        return {};
    if (D.size() > static_cast<size_t>(INT_MAX) ||
        D[0].size() > static_cast<size_t>(INT_MAX) ||
        B.size() < D.size())
        throw length_error("DupAggtree input shape exceeds supported range");

    const int n = static_cast<int>(D.size());
    const int dim = static_cast<int>(D[0].size());
    const size_t dimSize = static_cast<size_t>(dim);
    for (size_t i = 1; i < D.size(); ++i)
        if (D[i].size() != dimSize)
            throw invalid_argument("DupAggtree rows have different widths");

    int level = int(ceil(log2(max(n, 2)))) + 1;

    // Per-level sizes and flat offsets
    vector<int> lsz(level);
    vector<size_t> loff(level);
    size_t total = 0;
    lsz[0] = n;
    for (int h = 0; h < level; h++)
    {
        if (h > 0)
            lsz[h] = lsz[h - 1] / 2 + lsz[h - 1] % 2;
        loff[h] = total;
        total += static_cast<size_t>(lsz[h]);
    }

    // Single allocation for all tree-node data (no per-node vector)
    const size_t cells = checkedElementCount(total, dimSize);
    vector<int> A_px(cells, 0);
    vector<int> A_c  (total, 0);

    // Initialize level 0
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        const size_t row = static_cast<size_t>(i);
        const size_t base = (loff[0] + row) * dimSize;
        for (int j = 0; j < dim; j++)
            A_px[base + static_cast<size_t>(j)] = D[i][j];
        A_c[loff[0] + row] = B[i];
    }

    // Upstream phase
    for (int h = 1; h < level; h++)
    {
        int sz      = lsz[h];
        int sz_prev = lsz[h - 1];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < sz; i++)
        {
            const size_t row = static_cast<size_t>(i);
            const size_t left = 2 * row;
            const size_t t_base = (loff[h] + row) * dimSize;
            const size_t l_base = (loff[h - 1] + left) * dimSize;
            int l_c = A_c[loff[h - 1] + left];

            if (2 * i + 1 < sz_prev)
            {
                const size_t right = left + 1;
                const size_t r_base = (loff[h - 1] + right) * dimSize;
                int r_c = A_c[loff[h - 1] + right];
                A_c[loff[h] + row] = l_c & r_c;
                for (int j = 0; j < dim; j++)
                {
                    const size_t col = static_cast<size_t>(j);
                    A_px[t_base + col] = (!r_c) * A_px[r_base + col] + r_c * A_px[l_base + col];
                }
            }
            // else: A_c stays 0, A_px stays 0 (zero-initialized) — matches original
        }
    }

    // Downstream phase
    for (int h = level - 1; h > 0; h--)
    {
        int sz      = lsz[h];
        int sz_prev = lsz[h - 1];
        const size_t childSpan = static_cast<size_t>(1) << (h - 1);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < sz; i++)
        {
            const size_t row = static_cast<size_t>(i);
            const size_t left = 2 * row;
            const size_t par_base = (loff[h] + row) * dimSize;
            const size_t l_base = (loff[h - 1] + left) * dimSize;

            if (2 * i + 1 < sz_prev)
            {
                const size_t right = left + 1;
                const size_t r_base = (loff[h - 1] + right) * dimSize;
                // A_l for a tree node is always the B value of its first
                // leaf, so derive it directly instead of storing one int per
                // tree node.
                int r_l = B[right * childSpan];
                int l_c = A_c[loff[h - 1] + left];
                for (int j = 0; j < dim; j++)
                {
                    const size_t col = static_cast<size_t>(j);
                    // parent.lpx is the left child's upstream px.  At this
                    // point A_px[l_base] still holds that value; it is
                    // overwritten only by the memcpy below.
                    A_px[r_base + col] = r_l * (!l_c) * A_px[l_base + col]
                                       + r_l * l_c * A_px[par_base + col];
                }
            }
            // left.px = parent.px (done after right so the old left px is still available)
            memcpy(&A_px[l_base], &A_px[par_base], dimSize * sizeof(int));
        }
    }

    // The downstream pass no longer needs these arrays.  Release them before
    // materializing the result to reduce the live peak.
    vector<int>().swap(A_c);

    // Materialize in D itself.  Each cell reads its previous D value before
    // overwriting it, so this is equivalent to constructing a second Table.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        const size_t base = (loff[0] + static_cast<size_t>(i)) * dimSize;
        for (int j = 0; j < dim; j++)
            D[i][j] = B[i] * A_px[base + static_cast<size_t>(j)] + (1 - B[i]) * D[i][j];
    }
    return std::move(D);
}

// Flat-array variant: D_flat is row-major n×dim, result is row-major n×dim.
// Identical tree logic but operates on flat arrays throughout — avoids the
// O(n*dim) vector<vector<int>> copy that the regular variant performs.
vector<int> DupAggtree::DupAggtreeRunFlat()
{
    PrimitiveProfileScope primitiveScope(PrimitiveAggtree);
    if (!D_flat || !B_flat || n_rows_flat <= 0 || n_cols_flat <= 0)
        return {};

    const int n = n_rows_flat;
    const int dim = n_cols_flat;
    const size_t dimSize = static_cast<size_t>(dim);
    const size_t outputCells = checkedElementCount(static_cast<size_t>(n), dimSize);
    if (D_flat->size() < outputCells || B_flat->size() < static_cast<size_t>(n))
        throw invalid_argument("DupAggtree flat input is smaller than its declared shape");

    const int* D = D_flat->data();
    const int* B = B_flat->data();

    int level = int(ceil(log2(max(n, 2)))) + 1;

    vector<int> lsz(level);
    vector<size_t> loff(level);
    size_t total = 0;
    lsz[0] = n;
    for (int h = 0; h < level; h++)
    {
        if (h > 0)
            lsz[h] = lsz[h - 1] / 2 + lsz[h - 1] % 2;
        loff[h] = total;
        total += static_cast<size_t>(lsz[h]);
    }

    const size_t cells = checkedElementCount(total, dimSize);
    vector<int> A_px(cells, 0);
    vector<int> A_c  (total, 0);

    // Initialize level 0
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        const size_t row = static_cast<size_t>(i);
        const size_t base = (loff[0] + row) * dimSize;
        const size_t inputBase = row * dimSize;
        memcpy(&A_px[base], D + inputBase, dimSize * sizeof(int));
        A_c[loff[0] + row] = B[i];
    }

    // Upstream phase
    for (int h = 1; h < level; h++)
    {
        int sz      = lsz[h];
        int sz_prev = lsz[h - 1];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < sz; i++)
        {
            const size_t row = static_cast<size_t>(i);
            const size_t left = 2 * row;
            const size_t t_base = (loff[h] + row) * dimSize;
            const size_t l_base = (loff[h - 1] + left) * dimSize;
            int l_c = A_c[loff[h - 1] + left];

            if (2 * i + 1 < sz_prev)
            {
                const size_t right = left + 1;
                const size_t r_base = (loff[h - 1] + right) * dimSize;
                int r_c = A_c[loff[h - 1] + right];
                A_c[loff[h] + row] = l_c & r_c;
                for (int j = 0; j < dim; j++)
                {
                    const size_t col = static_cast<size_t>(j);
                    A_px[t_base + col] = (!r_c) * A_px[r_base + col] + r_c * A_px[l_base + col];
                }
            }
        }
    }

    // Downstream phase
    for (int h = level - 1; h > 0; h--)
    {
        int sz      = lsz[h];
        int sz_prev = lsz[h - 1];
        const size_t childSpan = static_cast<size_t>(1) << (h - 1);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < sz; i++)
        {
            const size_t row = static_cast<size_t>(i);
            const size_t left = 2 * row;
            const size_t par_base = (loff[h] + row) * dimSize;
            const size_t l_base = (loff[h - 1] + left) * dimSize;

            if (2 * i + 1 < sz_prev)
            {
                const size_t right = left + 1;
                const size_t r_base = (loff[h - 1] + right) * dimSize;
                int r_l = B[right * childSpan];
                int l_c = A_c[loff[h - 1] + left];
                for (int j = 0; j < dim; j++)
                {
                    const size_t col = static_cast<size_t>(j);
                    A_px[r_base + col] = r_l * (!l_c) * A_px[l_base + col]
                                       + r_l * l_c * A_px[par_base + col];
                }
            }
            memcpy(&A_px[l_base], &A_px[par_base], dimSize * sizeof(int));
        }
    }

    // Collect result (flat)
    vector<int> res(outputCells);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        const size_t rowBase = static_cast<size_t>(i) * dimSize;
        const size_t base = (loff[0] + static_cast<size_t>(i)) * dimSize;
        for (int j = 0; j < dim; j++)
        {
            const size_t col = static_cast<size_t>(j);
            res[rowBase + col] = B[i] * A_px[base + col] + (1 - B[i]) * D[rowBase + col];
        }
    }
    return res;
}
