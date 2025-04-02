#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>  // for remove_if, etc.

using namespace std;

// Particle definition: tag, charge, positions, momenta
struct Particle {
    int tag;
    double q;
    double x, y, z;
    double px, py, pz;
};

// Grid cell definition: Electric and Magnetic field components
struct Grid {
    double Ex, Ey, Ez;
    double Bx, By, Bz;
};

// Tile structure: contains particles, grid cells, neighbor info, etc.
struct Tile {
    int globalID;          // Global tile ID (unchanged even if tile moves)
    int tileRow, tileCol;  // Global tile row/col in the overall domain
    int nx, ny;            // Interior sizes
    int currentRank;       // Current rank that owns the tile

    vector<Particle> particles;
    vector<Grid> grid;
};

// RankInfo structure: which tiles are *currently* on this rank.
struct RankInfo {
    int rank, rankRow, rankCol;
    int tileRows, tileCols;
    // The list of tiles that belong to this rank *at the moment*:
    vector<Tile> tiles;
};

// Directions (left, right, up, down, top-left, top-right, bottom-left, bottom-right) (d = displacement)
static const int dRow[8] = { 0,  0, -1,  1, -1, -1,  1,  1 };
static const int dCol[8] = {-1,  1,  0,  0, -1,  1, -1,  1 };
static const int opposite[8] = { 1,  0,  3,  2,  7,  6,  5,  4 };

// Return a near-square decomposition (R rows x C columns) for 'size' total processes
void findBestGrid(int size, int &R, int &C) {
    R = static_cast<int>(std::sqrt((double)size));
    while (R > 1 && (size % R != 0)) {
        R--;
    }
    C = size / R;
}

// Return a near-square decomposition for 'numTiles' per rank
void findBestTileGrid(int numTiles, int &tileRows, int &tileCols) {
    tileRows = static_cast<int>(std::sqrt((double)numTiles));
    while (tileRows > 1 && (numTiles % tileRows != 0)) {
        tileRows--;
    }
    tileCols = numTiles / tileRows;
}

// Suppose the total domain of tiles is (globalTileRows x globalTileCols)
// Then a tile at (row, col) -> globalID = row * globalTileCols + col

// Given rankRow, tileRows, and local row tRow, produce the tile's global row
inline int tileGlobalRow(int rankRow, int tileRows, int tRow) {
    return rankRow * tileRows + tRow;
}

// Given rankCol, tileCols, and local column tCol, produce the tile's global col
inline int tileGlobalCol(int rankCol, int tileCols, int tCol) {
    return rankCol * tileCols + tCol;
}

// Convert (globalRow, globalCol) to a single integer ID in row-major order
inline int getGlobalID(int globalRow, int globalCol, int globalTileCols) {
    return globalRow * globalTileCols + globalCol;
}

// Recover (globalRow, globalCol) from the globalID
inline void getGlobalRowCol(int gID, int globalTileCols, int &row, int &col) {
    row = gID / globalTileCols;
    col = gID % globalTileCols;
}

// Return the globalID of the tile’s neighbor in direction d, with 2D periodic wrapping
int getNeighborGID(int myGID, int d, int globalTileRows, int globalTileCols) {
    int row, col;
    getGlobalRowCol(myGID, globalTileCols, row, col);
    int newRow = row + dRow[d];
    int newCol = col + dCol[d];

    // Wrap in row:
    if (newRow < 0)                   newRow = globalTileRows - 1;
    else if (newRow >= globalTileRows) newRow = 0;

    // Wrap in col:
    if (newCol < 0)                   newCol = globalTileCols - 1;
    else if (newCol >= globalTileCols) newCol = 0;

    return getGlobalID(newRow, newCol, globalTileCols);
}

// Given a globalID, return the local tile index within the rank
int getGlobalIDFromLocalFunc(int r, int localTile, int R, int C, int tileRows, int tileCols, int globalTileCols) {
    int rR = r / C;               // the rank’s row
    int rC = r % C;               // the rank’s col
    int localRow = localTile / tileCols;
    int localCol = localTile % tileCols;

    int gRow = rR * tileRows + localRow;
    int gCol = rC * tileCols + localCol;
    return getGlobalID(gRow, gCol, globalTileCols);
}

// Compute a unique tag for MPI based on (tileGID, direction)
inline int computeTag(int tileGID, int direction) {
    // tileGID * 8 + direction
    return tileGID * 8 + direction;
}

// Extract the guard region from “tile” in direction d into a buffer
vector<Grid> packSendBuffer(const Tile &tile, int d, int guard) {
    int totalX = tile.nx + 2 * guard;
    int totalY = tile.ny + 2 * guard;

    vector<Grid> sendBuffer;
    int sendSize = 0;

    if (d == 0) { // Left
        sendSize = tile.ny * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < tile.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 1) { // Right
        sendSize = tile.ny * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < tile.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + (tile.nx - guard) + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 2) { // Up
        sendSize = tile.nx * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.nx; col++) {
                int srcRow = guard + row;
                int srcCol = guard + col;
                sendBuffer[row * tile.nx + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 3) { // Down
        sendSize = tile.nx * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.nx; col++) {
                int srcRow = guard + (tile.ny - guard) + row;
                int srcCol = guard + col;
                sendBuffer[row * tile.nx + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 4) { // Top-left
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 5) { // Top-right
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + (tile.nx - guard) + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 6) { // Bottom-left
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + (tile.ny - guard) + row;
                int srcCol = guard + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 7) { // Bottom-right
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + (tile.ny - guard) + row;
                int srcCol = guard + (tile.nx - guard) + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }

    return sendBuffer;
}

// Update the guard region in “tile” in direction d from the receive buffer
void updateGuardRegion(Tile &tile, int d, int guard, const vector<Grid> &rbuf) {
    int totalX = tile.nx + 2 * guard;
    int totalY = tile.ny + 2 * guard;

    if (d == 0) { // Left
        for (int row = 0; row < tile.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int dstRow = guard + row;
                int dstCol = col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 1) { // Right
        for (int row = 0; row < tile.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int dstRow = guard + row;
                int dstCol = totalX - guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 2) { // Up
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.nx; col++) {
                int dstRow = row;
                int dstCol = guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * tile.nx + col];
            }
        }
    }
    else if (d == 3) { // Down
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.nx; col++) {
                int dstRow = totalY - guard + row;
                int dstCol = guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * tile.nx + col];
            }
        }
    }
    else if (d == 4) { // Top-left
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[row * totalX + col] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 5) { // Top-right
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[row * totalX + (totalX - guard + col)] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 6) { // Bottom-left
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[(totalY - guard + row) * totalX + col] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 7) { // Bottom-right
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[(totalY - guard + row) * totalX + (totalX - guard + col)] = rbuf[row * guard + col];
            }
        }
    }
}

// -------------------------------------- MAIN LOOP --------------------------------------
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute decomposition among ranks
    int R, C;
    findBestGrid(size, R, C);
    int rankRow = rank / C;
    int rankCol = rank % C;

    // Each rank *initially* has tileRows*tileCols = numTiles
    int numTiles = 9;  // 3x3 per rank
    int tileRows, tileCols;
    findBestTileGrid(numTiles, tileRows, tileCols);

    RankInfo info;
    info.rank = rank;
    info.rankRow = rankRow;
    info.rankCol = rankCol;
    info.tileRows = tileRows;
    info.tileCols = tileCols;
    info.tiles.resize(numTiles);

    // Total domain has (R * tileRows) x (C * tileCols) tiles
    int globalTileRows = R * tileRows;
    int globalTileCols = C * tileCols;
    int totalGlobalTiles = globalTileRows * globalTileCols;

    // We keep an array “owner[gID]” that says which rank currently owns tile with globalID = gID
    vector<int> owner;
    owner.resize(totalGlobalTiles, -1); // fill with invalid

    // Initialize tile data
    int interior_nx = 10;
    int interior_ny = 10;
    int guard = 2;

    // Initialize the local tiles and compute each tile's global ID.
    for (int t = 0; t < numTiles; t++) {
        Tile &tile = info.tiles[t];
        int tRow = t / tileCols;
        int tCol = t % tileCols;

        int gRow = tileGlobalRow(rankRow, tileRows, tRow);
        int gCol = tileGlobalCol(rankCol, tileCols, tCol);

        tile.globalID = getGlobalID(gRow, gCol, globalTileCols);
        tile.tileRow = gRow;
        tile.tileCol = gCol;
        tile.nx = interior_nx;
        tile.ny = interior_ny;
        tile.currentRank = rank; // Set current rank

        int totalX = interior_nx + 2 * guard;
        int totalY = interior_ny + 2 * guard;
        tile.grid.resize(totalX * totalY);

        double val = rank + 0.1 * t; // Example for easy identification
        for (int i = 0; i < (int)tile.grid.size(); i++) {
            tile.grid[i].Ex = val;
        }
        // Mark the owner
        owner[tile.globalID] = rank;
    }

    // Make sure every rank now has the same “owner[]” info (use MPI_MAX)
    MPI_Allreduce(MPI_IN_PLACE, &owner[0], totalGlobalTiles, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // ---------------------------------------------------------------------------------
    // *** Demonstration of moving tiles #3 from ranks 0,1,2 to rank 3 (if size==4) ***
    // ---------------------------------------------------------------------------------
    if (size == 4) {
        // 1) Each of ranks 0,1,2 sends tile #3 to rank 3
        if (rank == 0 || rank == 1 || rank == 2) {
            if ((int)info.tiles.size() > 3) {
                Tile movingTile = info.tiles[3];
                int dest = 3;

                cout << "Rank " << rank << " is sending Tile with globalID="
                     << movingTile.globalID << " (currentRank=" << movingTile.currentRank << ") to Rank " << dest << endl << flush;

                // Send the globalID
                MPI_Send(&movingTile.globalID, 1, MPI_INT, dest, 999, MPI_COMM_WORLD);

                // Send the entire tile.grid data
                int nGrid = (int)movingTile.grid.size();
                MPI_Send(&movingTile.grid[0],
                         nGrid * sizeof(Grid),
                         MPI_BYTE,
                         dest,
                         1000,
                         MPI_COMM_WORLD);

                // Mark new owner in external array
                owner[movingTile.globalID] = dest;

                // Remove it locally
                info.tiles.erase(info.tiles.begin() + 3);
            }
        }
        else if (rank == 3) {
            // We expect 3 incoming tiles (one from each of ranks 0,1,2).
            for (int source = 0; source < 3; source++) {
                int inGID; // Incoming globalID
                MPI_Recv(&inGID, 1, MPI_INT, source, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Create a tile
                Tile newTile;
                newTile.globalID = inGID;
                newTile.nx = interior_nx;
                newTile.ny = interior_ny;
                newTile.currentRank = rank;

                int totalX = interior_nx + 2 * guard;
                int totalY = interior_ny + 2 * guard;
                newTile.grid.resize(totalX * totalY);

                // Receive data
                MPI_Recv(&newTile.grid[0],
                         (int)newTile.grid.size() * sizeof(Grid),
                         MPI_BYTE,
                         source,
                         1000,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                // Find tileRow,tileCol from inGID
                {
                    int r_, c_;
                    getGlobalRowCol(inGID, globalTileCols, r_, c_);
                    newTile.tileRow = r_;
                    newTile.tileCol = c_;
                }

                // Mark in ownership array
                owner[inGID] = 3;

                // Add it to local tiles
                info.tiles.push_back(newTile);

                cout << "Rank " << rank << " has received Tile with globalID="
                     << inGID << " (currentRank=" << newTile.currentRank << ") from Rank " << source << endl << flush;
            }
        }

        // 2) Wait for all the tile migration to finish
        MPI_Barrier(MPI_COMM_WORLD);

        // 3) Update ownership array with MPI_MAX so that everyone sees the new owners
        MPI_Allreduce(MPI_IN_PLACE, &owner[0], totalGlobalTiles, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    // Communication of guard cells based on dynamic ownership
    int localTileCount = (int)info.tiles.size();
    int totalComm = localTileCount * 8;
    vector<MPI_Request> requests(totalComm * 2);
    int reqCount = 0;

    // Prepare storage for receives:
    vector< vector< vector<Grid> > > recvBuffers(localTileCount, vector< vector<Grid> >(8));

    // 1) Post all non-blocking receives
    for (int i = 0; i < localTileCount; i++) {
        Tile &tile = info.tiles[i];
        int gID = tile.globalID;

        for (int d = 0; d < 8; d++) {
            int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
            int nbrRank = owner[nbrGID];

            // The neighbor will send with tag = computeTag(nbrGID, opposite[d])
            int nbrDirection = opposite[d];
            int tag = computeTag(nbrGID, nbrDirection);

            int bufferSize = 0;
            if (d == 0 || d == 1)
                bufferSize = interior_ny * guard; // left/right
            else if (d == 2 || d == 3)
                bufferSize = interior_nx * guard; // up/down
            else
                bufferSize = guard * guard;       // corners

            recvBuffers[i][d].resize(bufferSize);

            MPI_Irecv(&recvBuffers[i][d][0],
                      bufferSize * sizeof(Grid),
                      MPI_BYTE,
                      nbrRank,
                      tag,
                      MPI_COMM_WORLD,
                      &requests[reqCount]);
            reqCount++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 2) Post all non-blocking sends
    for (int i = 0; i < localTileCount; i++) {
        Tile &tile = info.tiles[i];
        int gID = tile.globalID;

        for (int d = 0; d < 8; d++) {
            int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
            int nbrRank = owner[nbrGID];

            // Our send tag = computeTag(gID, d)
            int tagSend = computeTag(gID, d);

            vector<Grid> sendBuffer = packSendBuffer(tile, d, guard);

            MPI_Isend(&sendBuffer[0],
                      (int)sendBuffer.size() * sizeof(Grid),
                      MPI_BYTE,
                      nbrRank,
                      tagSend,
                      MPI_COMM_WORLD,
                      &requests[reqCount]);
            reqCount++;
        }
    }

    // 3) Wait for all requests
    MPI_Waitall(reqCount, &requests[0], MPI_STATUSES_IGNORE);

    // 4) Update guard regions
    for (int i = 0; i < localTileCount; i++) {
        Tile &tile = info.tiles[i];
        for (int d = 0; d < 8; d++) {
            updateGuardRegion(tile, d, guard, recvBuffers[i][d]);
        }
    }

    // Prints for debugging
    MPI_Barrier(MPI_COMM_WORLD);
    for (int p = 0; p < size; p++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            cout << "\n===== OUTPUT FOR RANK " << rank
                 << " (Rank coords: " << rankRow << "," << rankCol << ") =====\n";

            for (int i = 0; i < (int)info.tiles.size(); i++) {
                Tile &tile = info.tiles[i];
                int totalX = tile.nx + 2 * guard;
                int totalY = tile.ny + 2 * guard;

                cout << "Tile GID=" << tile.globalID
                     << " (Row=" << tile.tileRow
                     << ", Col=" << tile.tileCol
                     << ", currentRank=" << tile.currentRank << ") \n"; // CHANGED: print currentRank

                // Left, Right, Top, Bottom
                cout << "  Left Guard (first 2 rows):\n    ";
                for (int row = guard; row < guard + 2 && row < (tile.ny + guard); row++) {
                    for (int col = 0; col < guard; col++) {
                        cout << tile.grid[row * totalX + col].Ex << " ";
                    }
                    cout << " | ";
                }
                cout << "\n  Right Guard (first 2 rows):\n    ";
                for (int row = guard; row < guard + 2 && row < (tile.ny + guard); row++) {
                    for (int col = totalX - guard; col < totalX; col++) {
                        cout << tile.grid[row * totalX + col].Ex << " ";
                    }
                    cout << " | ";
                }
                cout << "\n  Top Guard (first 2 cols):\n    ";
                for (int row = 0; row < guard; row++) {
                    for (int col = guard; col < guard + 2 && col < (tile.nx + guard); col++) {
                        cout << tile.grid[row * totalX + col].Ex << " ";
                    }
                    cout << " | ";
                }
                cout << "\n  Bottom Guard (first 2 cols):\n    ";
                for (int row = totalY - guard; row < totalY - guard + 2; row++) {
                    for (int col = guard; col < guard + 2 && col < (tile.nx + guard); col++) {
                        cout << tile.grid[row * totalX + col].Ex << " ";
                    }
                    cout << " | ";
                }

                // Oblique corners
                cout << "\n  Top-Left Guard:\n    ";
                for (int row = 0; row < guard; row++) {
                    for (int col = 0; col < guard; col++) {
                        cout << tile.grid[row * totalX + col].Ex << " ";
                    }
                    cout << " | ";
                }
                cout << "\n  Top-Right Guard:\n    ";
                for (int row = 0; row < guard; row++) {
                    for (int col = totalX - guard; col < totalX; col++) {
                        cout << tile.grid[row * totalX + col].Ex << " ";
                    }
                    cout << " | ";
                }
                cout << "\n  Bottom-Left Guard:\n    ";
                for (int row = totalY - guard; row < totalY; row++) {
                    for (int col = 0; col < guard; col++) {
                        cout << tile.grid[row * totalX + col].Ex << " ";
                    }
                    cout << " | ";
                }
                cout << "\n  Bottom-Right Guard:\n    ";
                for (int row = totalY - guard; row < totalY; row++) {
                    for (int col = totalX - guard; col < totalX; col++) {
                        cout << tile.grid[row * totalX + col].Ex << " ";
                    }
                    cout << " | ";
                }
                cout << "\n";
            }
            cout.flush();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
