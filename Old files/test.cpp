#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>  // *** NEW DEBUG CODE ***
#include <cstdlib>        // for fabs
#include "Auxiliar_functions.h"  // Contains all helper functions and their definitions
#include "Field_update.h"        // External field update routines for evolving electromagnetic fields
#include "HDF5_output.h"         // Function to save tile data to an HDF5 file

using namespace std;

// *** NEW DEBUG CODE ***
// This function compares the guard cell region of each local tile with the corresponding interior region
// of its neighbor tile in the specified direction. For now we implement for four directions:
// d=0: left, d=1: right, d=2: up, d=3: down.
// (The conventions here follow those used in your packSendBuffer/updateGuardRegion routines.)
// globalTileRows and globalTileCols are needed for computing neighbor global IDs.
void checkTileGuardConsistency(const RankInfo &info, 
                               const vector<int> &owner, 
                               int guard,
                               int globalTileRows,
                               int globalTileCols) {
    // Build a mapping from globalID to pointer to local tile.
    unordered_map<int, const Tile*> localTiles;
    for (const auto &tile : info.tiles) {
        // Only include if this tile is local.
        if(owner[tile.info.globalID] == info.rank)
            localTiles[tile.info.globalID] = &tile;
    }
    
    // For each local tile, check the neighbor in each direction (0: left, 1: right, 2: up, 3: down).
    for (const auto &kv : localTiles) {
        int gid = kv.first;
        const Tile* tile = kv.second;
        int totalX = tile->info.nx + 2 * guard;
        int totalY = tile->info.ny + 2 * guard;
        
        // For each direction, we will compute the maximum difference
        // between the receiving tile's guard region and the corresponding interior region of the neighbor.
        // (We assume the neighbor's interior region is already computed correctly.)
        for (int d = 0; d < 4; d++) {
            int nbrGID = getNeighborGID(gid, d, globalTileRows, globalTileCols);
            // Only compare if the neighbor tile is local.
            if (localTiles.find(nbrGID) == localTiles.end())
                continue;
            const Tile* nbrTile = localTiles[nbrGID];
            int nbrTotalX = nbrTile->info.nx + 2 * guard;
            int nbrTotalY = nbrTile->info.ny + 2 * guard;
            double maxDiff = 0.0;
            
            // For left (d==0): current tile receives data in its left guard region.
            // The expected data come from the neighbor tile's rightmost interior columns.
            if (d == 0) {
                // For each row in the interior (rows guard ... guard+ny-1)
                for (int r = guard; r < guard + tile->info.ny; r++) {
                    // Current tile left guard region: columns 0 to guard-1.
                    // Neighbor's interior: columns [guard, guard+nx-1].
                    // Its right boundary is columns from (guard + nbrTile->info.nx - guard) to (guard + nbrTile->info.nx - 1)
                    for (int c = 0; c < guard; c++) {
                        int idxTile = r * totalX + c;
                        int idxNbr = r * nbrTotalX + (guard + nbrTile->info.nx - guard + c);
                        double diff = fabs(tile->grid[idxTile].Bz - nbrTile->grid[idxNbr].Bz);
                        if (diff > maxDiff)
                            maxDiff = diff;
                    }
                }
                cout << "Tile " << gid << " (left guard) vs neighbor " << nbrGID 
                     << " (right interior) max diff: " << maxDiff << endl;
            }
            // For right (d==1): current tile receives in its right guard region.
            else if (d == 1) {
                for (int r = guard; r < guard + tile->info.ny; r++) {
                    for (int c = tile->info.nx + guard; c < tile->info.nx + 2 * guard; c++) {
                        int idxTile = r * totalX + c;
                        // Expected data from neighbor's left boundary of its interior:
                        int idxNbr = r * nbrTotalX + guard + (c - (tile->info.nx + guard));
                        double diff = fabs(tile->grid[idxTile].Bz - nbrTile->grid[idxNbr].Bz);
                        if (diff > maxDiff)
                            maxDiff = diff;
                    }
                }
                cout << "Tile " << gid << " (right guard) vs neighbor " << nbrGID 
                     << " (left interior) max diff: " << maxDiff << endl;
            }
            // For up (d==2): current tile receives in its top guard region.
            else if (d == 2) {
                for (int c = guard; c < guard + tile->info.nx; c++) {
                    for (int r = 0; r < guard; r++) {
                        int idxTile = r * totalX + c;
                        // Expected data from neighbor's bottom interior: rows from (guard + nbrTile->info.ny - guard) to (guard + nbrTile->info.ny - 1)
                        int idxNbr = (guard + nbrTile->info.ny - guard + r) * nbrTotalX + c;
                        double diff = fabs(tile->grid[idxTile].Ey - nbrTile->grid[idxNbr].Ey);
                        if (diff > maxDiff)
                            maxDiff = diff;
                    }
                }
                cout << "Tile " << gid << " (top guard) vs neighbor " << nbrGID 
                     << " (bottom interior) max diff: " << maxDiff << endl;
            }
            // For down (d==3): current tile receives in its bottom guard region.
            else if (d == 3) {
                for (int c = guard; c < guard + tile->info.nx; c++) {
                    for (int r = tile->info.ny + guard; r < tile->info.ny + 2 * guard; r++) {
                        int idxTile = r * totalX + c;
                        // Expected data from neighbor's top interior: rows from guard to guard+guard-1.
                        int idxNbr = (guard + (r - (tile->info.ny + guard))) * nbrTotalX + c;
                        double diff = fabs(tile->grid[idxTile].Ey - nbrTile->grid[idxNbr].Ey);
                        if (diff > maxDiff)
                            maxDiff = diff;
                    }
                }
                cout << "Tile " << gid << " (bottom guard) vs neighbor " << nbrGID 
                     << " (top interior) max diff: " << maxDiff << endl;
            }
        }
    }
}
// *** END NEW DEBUG CODE ***

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine grid decomposition for the given number of ranks
    int R, C; // Number of rows (R) and columns (C) in the rank grid
    findBestGrid(size, R, C);
    int rankRow = rank / C;
    int rankCol = rank % C;

    // Each rank initially holds 9 tiles (arranged in a 3x3 grid)
    int numTiles = 9;
    int tileRows, tileCols;
    findBestTileGrid(numTiles, tileRows, tileCols);

    // Initialize the RankInfo for the current rank
    RankInfo info;
    info.rank = rank;
    info.rankRow = rankRow;
    info.rankCol = rankCol;
    info.tileRows = tileRows;
    info.tileCols = tileCols;
    info.tiles.resize(numTiles);

    // Total global domain of tiles is (R * tileRows) x (C * tileCols)
    int globalTileRows = R * tileRows;
    int globalTileCols = C * tileCols;
    int totalGlobalTiles = globalTileRows * globalTileCols;

    // Ownership array: which rank owns each tile (initially, each tile is owned by its local rank)
    vector<int> owner(totalGlobalTiles, -1);

    // Simulation Setup
    double box_x = 10.0, box_y = 10.0;                  // Physical box dimensions [c/ω_p]
    int nx = 180, ny = 180;                             // Total number of grid cells in x and y
    double dx = box_x / nx;                             // Spatial step size in x [c/ω_p]
    double dy = box_y / ny;                             // Spatial step size in y [c/ω_p]
    int guard = 2;                                      // Number of guard cells

    int interior_nx = nx / (C * tileCols);              // Number of interior cells in x for each tile
    int interior_ny = ny / (R * tileRows);              // Number of interior cells in y for each tile

    double sim_time = 0.1;                              // Total simulation time [ω_p⁻¹]
    double dt_courant = 1/(1/pow(dx,2)+(1/pow(dy,2)));  // Courant Condition
    double dt = 0.99 * dt_courant;                      // Time step size (respecting CFL)
    int total_steps = static_cast<int>(sim_time / dt);  // Number of simulation steps
    int save_frequency = 10;                            // Frequency of saving simulation data

    // Initialize each tile on this rank
    for (int t = 0; t < numTiles; t++) {
        Tile &tile = info.tiles[t];
        int tRow = t / tileCols; // Tile row index within the rank
        int tCol = t % tileCols; // Tile column index within the rank
        int gRow = tileGlobalRow(rankRow, tileRows, tRow); // Global row index of the tile
        int gCol = tileGlobalCol(rankCol, tileCols, tCol); // Global column index of the tile

        tile.info.globalID = getGlobalID(gRow, gCol, globalTileCols);
        tile.info.tileRow = gRow;
        tile.info.tileCol = gCol;
        tile.info.nx = interior_nx;
        tile.info.ny = interior_ny;
        tile.info.currentRank = rank;

        int totalX = interior_nx + 2 * guard; 
        int totalY = interior_ny + 2 * guard;
        tile.grid.resize(totalX * totalY);

        // Compute global offsets for each tile
        int global_x_offset = tile.info.tileCol * interior_nx;
        int global_y_offset = (globalTileRows - 1 - tile.info.tileRow) * interior_ny;

        // Initialize the fields for each tile
        double A = 0.1;                     // Wave amplitude
        double kx = 5 * 2 * M_PI / box_x;   // Wave number in x
        double ky = 0;                      // Wave number in y
        for (int j = 0; j < totalY; j++) {
            for (int i = 0; i < totalX; i++) {
                double x_Ey = (global_x_offset + i - guard) * dx;
                double y_Ey = (global_y_offset + j - guard + 0.5) * dy;
                double x_Bz = (global_x_offset + i - guard + 0.5) * dx;
                double y_Bz = (global_y_offset + j - guard + 0.5) * dy;
                int idx = j * totalX + i;
                tile.grid[idx].Ex = 0.0;
                tile.grid[idx].Ey = A * sin(kx * x_Ey);
                tile.grid[idx].Ez = 0.0;
                tile.grid[idx].Bx = 0.0;
                tile.grid[idx].By = 0.0;
                tile.grid[idx].Bz = A * sin(kx * x_Bz);
            }
        }
        owner[tile.info.globalID] = rank;
    }

    // Synchronize ownership info across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &owner[0], totalGlobalTiles, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Save initial state (step 0)
    std::string filename = "Simulation/Fields/fields_rank_" + to_string(rank) + "_step_0.h5";
    saveRankData(info, guard, filename);

    // ----------------------- Time-Stepping Simulation Loop -----------------------
    for (int step = 1; step < total_steps; step++) { // Start at step 1
        // --- Phase A: Update B Field ---
        for (int i = 0; i < info.tiles.size(); i++) {
            Tile &tile = info.tiles[i];
            int totalX = tile.info.nx + 2 * guard;
            int totalY = tile.info.ny + 2 * guard;
            int totalCells = totalX * totalY;
            vector<GridE> Efull(totalCells);
            vector<GridB> BhalfOld(totalCells);
            for (int idx = 0; idx < totalCells; idx++) {
                Efull[idx].Ex = tile.grid[idx].Ex;
                Efull[idx].Ey = tile.grid[idx].Ey;
                Efull[idx].Ez = tile.grid[idx].Ez;
                BhalfOld[idx].Bx = tile.grid[idx].Bx;
                BhalfOld[idx].By = tile.grid[idx].By;
                BhalfOld[idx].Bz = tile.grid[idx].Bz;
            }
            vector<GridB> BhalfNew(totalCells);
            updateBhalf(Efull, BhalfOld, BhalfNew, interior_nx, interior_ny, guard, dt, dx, dy);
            for (int idx = 0; idx < totalCells; idx++) {
                tile.grid[idx].Bx = BhalfNew[idx].Bx;
                tile.grid[idx].By = BhalfNew[idx].By;
                tile.grid[idx].Bz = BhalfNew[idx].Bz;
            }
        }

        // --- Phase B: Guard-cell Communication after B Update ---
        {
            int localTileCount = info.tiles.size();
            int totalComm = localTileCount * 8;
            vector<MPI_Request> requests(totalComm * 2);
            int reqCount = 0;
            vector< vector< vector<Grid> > > recvBuffers(localTileCount, vector< vector<Grid> >(8));
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    int nbrDirection = opposite[d];
                    int tag = computeTag(nbrGID, nbrDirection);
                    int bufferSize = 0;
                    if (d == 0 || d == 1)
                        bufferSize = interior_ny * guard;
                    else if (d == 2 || d == 3)
                        bufferSize = interior_nx * guard;
                    else
                        bufferSize = guard * guard;
                    recvBuffers[i][d].resize(bufferSize);
                    MPI_Irecv(&recvBuffers[i][d][0], bufferSize * sizeof(Grid), MPI_BYTE, nbrRank, tag, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    int tagSend = computeTag(gID, d);
                    vector<Grid> sendBuffer = packSendBuffer(tile, d, guard);
                    MPI_Isend(&sendBuffer[0], sendBuffer.size() * sizeof(Grid), MPI_BYTE, nbrRank, tagSend, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            MPI_Waitall(reqCount, &requests[0], MPI_STATUSES_IGNORE);
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                for (int d = 0; d < 8; d++) {
                    updateGuardRegion(tile, d, guard, recvBuffers[i][d]);
                }
            }
        } // End Phase B

        // *** NEW DEBUG CODE ***: Compare guard cells with corresponding neighbor interiors
        // This function will print (for each local tile, if the neighbor is also local)
        // the maximum difference between the guard region and the corresponding interior region.
        checkTileGuardConsistency(info, owner, guard, globalTileRows, globalTileCols);
        // *** END NEW DEBUG CODE ***

        // --- Phase C: Update E Field ---
        for (int i = 0; i < info.tiles.size(); i++) {
            Tile &tile = info.tiles[i];
            int totalX = tile.info.nx + 2 * guard;
            int totalY = tile.info.ny + 2 * guard;
            int totalCells = totalX * totalY;
            vector<GridE> Efull(totalCells);
            for (int idx = 0; idx < totalCells; idx++) {
                Efull[idx].Ex = tile.grid[idx].Ex;
                Efull[idx].Ey = tile.grid[idx].Ey;
                Efull[idx].Ez = tile.grid[idx].Ez;
            }
            vector<GridB> Bhalf(totalCells);
            for (int idx = 0; idx < totalCells; idx++) {
                Bhalf[idx].Bx = tile.grid[idx].Bx;
                Bhalf[idx].By = tile.grid[idx].By;
                Bhalf[idx].Bz = tile.grid[idx].Bz;
            }
            updateEfull(Efull, Bhalf, interior_nx, interior_ny, guard, dt, dx, dy);
            for (int idx = 0; idx < totalCells; idx++) {
                tile.grid[idx].Ex = Efull[idx].Ex;
                tile.grid[idx].Ey = Efull[idx].Ey;
                tile.grid[idx].Ez = Efull[idx].Ez;
            }
        }

        // --- Phase D: Guard-cell Communication after E Update ---
        {
            int localTileCount = info.tiles.size();
            int totalComm = localTileCount * 8;
            vector<MPI_Request> requests(totalComm * 2);
            int reqCount = 0;
            vector< vector< vector<Grid> > > recvBuffers(localTileCount, vector< vector<Grid> >(8));
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    int nbrDirection = opposite[d];
                    int tag = computeTag(nbrGID, nbrDirection);
                    int bufferSize = 0;
                    if (d == 0 || d == 1)
                        bufferSize = interior_ny * guard;
                    else if (d == 2 || d == 3)
                        bufferSize = interior_nx * guard;
                    else
                        bufferSize = guard * guard;
                    recvBuffers[i][d].resize(bufferSize);
                    MPI_Irecv(&recvBuffers[i][d][0], bufferSize * sizeof(Grid), MPI_BYTE, nbrRank, tag, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    int tagSend = computeTag(gID, d);
                    vector<Grid> sendBuffer = packSendBuffer(tile, d, guard);
                    MPI_Isend(&sendBuffer[0], sendBuffer.size() * sizeof(Grid), MPI_BYTE, nbrRank, tagSend, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            MPI_Waitall(reqCount, &requests[0], MPI_STATUSES_IGNORE);
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                for (int d = 0; d < 8; d++) {
                    updateGuardRegion(tile, d, guard, recvBuffers[i][d]);
                }
            }
        } // End Phase D

        // --- Phase E: Dynamic Tile Migration (unchanged) ---
        if (step == 5 && size == 4) {
            if (rank == 0 || rank == 1 || rank == 2) {
                if (info.tiles.size() > 3) {
                    removeTileFromRank(info, owner, 3, 3);
                }
            } else if (rank == 3) {
                for (int source = 0; source < 3; source++) {
                    addTileToRank(info, owner, source, globalTileCols, interior_nx, interior_ny, guard);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &owner[0], totalGlobalTiles, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }

        // --- Phase F: Save Simulation Data ---
        if (step % save_frequency == 0) {
            std::string filename = "Simulation/Fields/fields_rank_" + to_string(rank) + "_step_" + to_string(step) + ".h5";
            saveRankData(info, guard, filename);
        }
    } // End simulation loop

    // Final Output: Print tile information after simulation completes
    MPI_Barrier(MPI_COMM_WORLD);
    for (int p = 0; p < size; p++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            cout << "\n===== FINAL OUTPUT FOR RANK " << rank
                 << " (Rank coords: " << rankRow << "," << rankCol
                 << ") after " << total_steps << " steps =====\n";
            for (int i = 0; i < info.tiles.size(); i++) {
                Tile &tile = info.tiles[i];
                cout << "Tile GID=" << tile.info.globalID
                     << " (Row=" << tile.info.tileRow
                     << ", Col=" << tile.info.tileCol
                     << ", currentRank=" << tile.info.currentRank << ")\n";
            }
            cout.flush();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "\nSimulation complete.\n";
    }
    MPI_Finalize();
    return 0;
}
