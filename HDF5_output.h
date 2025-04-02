#ifndef HDF5_OUTPUT_H
#define HDF5_OUTPUT_H

#include <string>
#include "Auxiliar_functions.h"  // Assumes definitions for RankInfo, Tile, and Grid
#include "H5Cpp.h"

// Save the field data for all tiles in the current rank to an HDF5 file
// filename: Name of the output file (e.g., "fields_rank_0.h5")
// guard: Number of guard cells (used to compute the tile dimensions)
void saveRankData(const RankInfo &info, int guard, const std::string &filename);

#endif // HDF5_OUTPUT_H
