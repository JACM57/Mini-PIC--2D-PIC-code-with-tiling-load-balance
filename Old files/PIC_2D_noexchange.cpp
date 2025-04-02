#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <utility> // for std::pair

using namespace std;

// Particle definition: tag, charge, positions (x,y,z) and momenta (px,py,pz)
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

// Tile structure: contains particles, grid cells, and neighbor information
struct Tile {
    vector<Particle> particles;  // Vector of particles
    vector<Grid> grid;           // 1D vector of grid cells, ordered in row-major format
    int tileRow, tileCol;        // Position in the local tile grid
    int neighborRank[8];         // Info about its eight rank neighbors
    int neighborTile[8];         // Info about its eight tile neighbors
    int nx, ny;
};

// RankInfo structure: contains MPI grid information and the list of tiles owned by this rank
struct RankInfo {
    int rank, rankRow, rankCol;  // MPI rank and its position in the global rank grid
    int tileRows, tileCols;      // Number of tiles in each row and column on this rank
    vector<Tile> tiles;          // The tiles on this rank
};


// Find a near-square decomposition of 'size' processes
void findBestGrid(int size, int &R, int &C) {
    // Start with the integer square root of the total processes for rows (R) and adjusts R until it divides size evenly
    // The number of columns (C) is then calculated as size / R
    // Example: size = 14, sqrt(14) ≈ 3.74 -> R = 3, 14 % 3 != 0 -> R = 2, 14 % 2 == 0 -> C = 7
    R = static_cast<int>(sqrt((double)size));
    while (R > 1 && (size % R != 0)) {
        R--;
    }
    C = size / R;
}

// Find a near-square decomposition of 'numTiles' (same as the function above but for tiles)
void findBestTileGrid(int numTiles, int &tileRows, int &tileCols) {
    tileRows = static_cast<int>(sqrt((double)numTiles));
    while (tileRows > 1 && (numTiles % tileRows != 0)) {
        tileRows--;
    }
    tileCols = numTiles / tileRows;
}


// Given a tile's row and col and its MPI rank's position and the grid dimensions,
// compute the neighbor tile's MPI rank and local tile index using periodic boundaries
pair<int,int> getNeighbor2D(
    int tRow, int tCol,          // Current tile position within a rank
    int rankRow, int rankCol,    // Rank position in the MPI grid
    int tileRows, int tileCols,  // Local grid dimensions
    int R, int C,                // Global MPI grid dimensions (R rows and C columns)
    int dRow, int dCol           // Displacement -> specify the direction of the neighbor relative to the current tile
    // Example: dRow = 0, dCol = -1 -> neighbor is to the left
) {
    int newTrow = tRow + dRow;
    int newTcol = tCol + dCol;
    // By default assume that the neighbor is on the same MPI rank (adjust if it crosses rank boundaries)
    int newRankRow = rankRow;
    int newRankCol = rankCol;

    // Wrap vertically
    if (newTrow < 0) { // The neighbor is above the top of the tile grid
        newTrow = tileRows - 1; // Wrap around to the bottom
        newRankRow = (rankRow == 0) ? R - 1 : rankRow - 1; 
    } else if (newTrow >= tileRows) { // The neighbor is below the bottom of the tile grid
        newTrow = 0; // Wrap around to the top
        newRankRow = (rankRow == R - 1) ? 0 : rankRow + 1;
    }
    // Wrap horizontally
    if (newTcol < 0) { // The neighbor is to the left of the tile grid
        newTcol = tileCols - 1; // Wrap around to the right
        newRankCol = (rankCol == 0) ? C - 1 : rankCol - 1;
    } else if (newTcol >= tileCols) { // The neighbor is to the right of the tile grid
        newTcol = 0; // Wrap around to the left
        newRankCol = (rankCol == C - 1) ? 0 : rankCol + 1;
    }
    // Calculate the new rank and tile index for the neighbor in row-major order
    int neighborRank = newRankRow * C + newRankCol;
    int neighborTile = newTrow * tileCols + newTcol;
    return make_pair(neighborRank, neighborTile);
}

// Ensure that when 1 tile sends a message in 1 direction, the receiving tile knows exactly which tag to expect based on the opposite direction
// 0=Left, 1=Right, 2=Up, 3=Down, 4=TopLeft, 5=TopRight, 6=BottomLeft, 7=BottomRight
// Tile expects data from left neighbor (d = 0) --> left neighbor will send its data in its right direction (d = 1) --> opposite[0] = 1
static const int opposite[8] = { 1, 0, 3, 2, 7, 6, 5, 4 };

// Compute a unique tag based on sender information and direction
// We assume each rank has the same number of tiles ---------------> ASK THE PROFESSOR IF THIS IS OKAY (I DONT THINK IT IS)
inline int computeTag(int senderRank, int senderTile, int direction, int tilesPerRank) {
    // senderRank * tilesPerRank + senderTile --> global tile index
    // Example: senderRank = 2, tilesPerRank = 9, senderTile = 4 --> 22 * 8 --> reserve tags 176 through 183
    // direction = 3 -> tag = 179
    return (senderRank * tilesPerRank + senderTile) * 8 + direction;
}

// packSendBuffer: Extracts the interior guards cells of a tile into a send buffer, depending on the direction d 
vector<Grid> packSendBuffer(const Tile &tile, int d, int guard) {
    int totalX = tile.nx + 2 * guard; // Will be used in row-major calculations
    // tile.nx and tile.ny are the interior dimensions
    vector<Grid> sendBuffer;
    int sendSize = 0;
    if (d == 0) { // Left --> 'guard' collumns
        sendSize = tile.ny * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < tile.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row; // Source row
                int srcCol = guard + col; // Source column
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    } else if (d == 1) { // Right --> 'guard' collumns
        sendSize = tile.ny * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < tile.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + (tile.nx - guard) + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    } else if (d == 2) { // Up --> 'guard' rows
        sendSize = tile.nx * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.nx; col++) {
                int srcRow = guard + row;
                int srcCol = guard + col;
                sendBuffer[row * tile.nx + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    } else if (d == 3) { // Down --> 'guard' rows
        sendSize = tile.nx * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.nx; col++) {
                int srcRow = guard + (tile.ny - guard) + row;
                int srcCol = guard + col;
                sendBuffer[row * tile.nx + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    } else if (d == 4) { // Top-left --> 'guard' * 'guard' square
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    } else if (d == 5) { // Top-right --> 'guard' * 'guard' square
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + (tile.nx - guard) + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    } else if (d == 6) { // Bottom-left --> 'guard' * 'guard' square
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + (tile.ny - guard) + row;
                int srcCol = guard + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    } else if (d == 7) { // Bottom-right --> 'guard' * 'guard' square
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


// updateGuardRegion: Updates a tile's guard cells for direction d using rbuf (receive buffer)
void updateGuardRegion(Tile &tile, int d, int guard, const vector<Grid>& rbuf) {
    int totalX = tile.nx + 2 * guard;
    int totalY = tile.ny + 2 * guard;
    if (d == 0) { // Left guard: rows [guard, guard+ny-1], columns [0, guard-1]
        for (int row = 0; row < tile.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int dstRow = guard + row; // Destination row
                int dstCol = col;         // Destination column
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * guard + col];
            }
        }
    } else if (d == 1) { // Right guard: rows [guard, guard+ny-1], columns [totalX-guard, totalX-1]
        for (int row = 0; row < tile.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int dstRow = guard + row;
                int dstCol = totalX - guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * guard + col];
            }
        }
    } else if (d == 2) { // Top guard: rows [0, guard-1], columns [guard, guard+nx-1]
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.nx; col++) {
                int dstRow = row;
                int dstCol = guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * tile.nx + col];
            }
        }
    } else if (d == 3) { // Bottom guard: rows [totalY-guard, totalY-1], columns [guard, guard+nx-1]
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.nx; col++) {
                int dstRow = totalY - guard + row;
                int dstCol = guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * tile.nx + col];
            }
        }
    } else if (d == 4) { // Top-left corner: rows [0, guard-1], columns [0, guard-1]
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[row * totalX + col] = rbuf[row * guard + col];
            }
        }
    } else if (d == 5) { // Top-right corner: rows [0, guard-1], columns [totalX-guard, totalX-1]
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[row * totalX + (totalX - guard + col)] = rbuf[row * guard + col];
            }
        }
    } else if (d == 6) { // Bottom-left corner: rows [totalY-guard, totalY-1], columns [0, guard-1]
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[(totalY - guard + row) * totalX + col] = rbuf[row * guard + col];
            }
        }
    } else if (d == 7) { // Bottom-right corner: rows [totalY-guard, totalY-1], columns [totalX-guard, totalX-1]
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[(totalY - guard + row) * totalX + (totalX - guard + col)] = rbuf[row * guard + col];
            }
        }
    }
}


// ------------------------------------------------------------ MAIN ------------------------------------------------------------
int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Set up a 2D grid of MPI ranks (R rows and C columns)
    int R, C;
    findBestGrid(size, R, C);
    int rankRow = rank / C; // Use integer division so that each row contains C ranks
    int rankCol = rank % C; // Example: if C=3, ranks 0,1,2 are in row 0, ranks 3,4,5 are in row 1, ...

    // Step 2: Each rank contains 9 tiles (3x3 grid calculated by findBestTileGrid)
    int numTiles = 9; // ---------------------------------------------------------------------> ASK THE PROFESSOR IF THIS IS OKAY
    int tileRows, tileCols;
    findBestTileGrid(numTiles, tileRows, tileCols);

    // Store information about the current rank and its tiles
    RankInfo info;
    info.rank = rank;
    info.rankRow = rankRow;
    info.rankCol = rankCol;
    info.tileRows = tileRows;
    info.tileCols = tileCols;
    info.tiles.resize(numTiles);

    // Step 3: Define interior dimensions and guard cells for each tile
    int interior_nx = 10;  // number of interior columns
    int interior_ny = 10;  // number of interior rows
    int guard = 2;         // number of guard cells on each side

    // Initialize each tile: assign indices, interior dims, allocate grid, and set test value (val)
    for (int t = 0; t < numTiles; t++) {
        Tile &tile = info.tiles[t];
        tile.tileRow = t / tileCols; // Tile row within the rank
        tile.tileCol = t % tileCols; // Tile column within the rank
        tile.nx = interior_nx;
        tile.ny = interior_ny;
        int totalX = interior_nx + 2 * guard; // Total columns (including guard cells)
        int totalY = interior_ny + 2 * guard; // Total rows (including guard cells)
        tile.grid.resize(totalX * totalY);
        double val = rank + 0.1 * t; // unique value per tile
        for (int i = 0; i < (int)tile.grid.size(); i++) {
            tile.grid[i].Ex = val;
        }
    }

    // Step 4: Determine the eight neighbors for each tile
    // Order: left, right, up, down, top-left, top-right, bottom-left, bottom-right
    static const int dRow[8] = { 0, 0, -1, 1, -1, -1, 1, 1 };
    static const int dCol[8] = { -1, 1, 0, 0, -1, 1, -1, 1 };
    for (int t = 0; t < numTiles; t++) {
        int tRow = info.tiles[t].tileRow; // Tile row within the rank
        int tCol = info.tiles[t].tileCol; // Tile column within the rank
        pair<int, int> nb;
        for (int d = 0; d < 8; d++) {
            nb = getNeighbor2D(tRow, tCol, rankRow, rankCol, // Get neighbor (nb) rank and tile index
                               tileRows, tileCols, R, C,
                               dRow[d], dCol[d]);
            info.tiles[t].neighborRank[d] = nb.first;  // Store neighbor rank
            info.tiles[t].neighborTile[d] = nb.second; // Store neighbor tile index
        }
    }

    // Step 5: Communication
    // Each tile exchanges its interior boundary with its neighbor in each of the 8 directions
    // Buffer sizes:
    //  - Horizontal (Left/Right): guard * interior_ny
    //  - Vertical (Up/Down): guard * interior_nx
    //  - Corners (Oblique): guard * guard
    int tilesPerRank = numTiles;
    int totalComm = numTiles * 8; // 8 directions per tile
    vector<MPI_Request> requests(totalComm * 2); // 1 send and 1 recv per direction
    int reqCount = 0; // Number of active requests

    // Store receive buffers as: recvBuffers[t][direction] is a vector<Grid>
    vector< vector< vector<Grid> > > recvBuffers(numTiles, vector< vector<Grid> >(8));

    // Step 5A: Post all non-blocking receives
    for (int t = 0; t < numTiles; t++) {
        Tile &tile = info.tiles[t];
        for (int d = 0; d < 8; d++) {
            int nbrRank = tile.neighborRank[d];
            int nbrTile = tile.neighborTile[d];
            int nbrDirection = opposite[d]; // neighbor's send direction
            int tag = computeTag(nbrRank, nbrTile, nbrDirection, tilesPerRank);
            int bufferSize = 0; // Size of the receive buffer
            if (d == 0 || d == 1)
                bufferSize = interior_ny * guard; // Left or Right
            else if (d == 2 || d == 3)
                bufferSize = interior_nx * guard; // Up or Down
            else
                bufferSize = guard * guard; // Oblique
            recvBuffers[t][d].resize(bufferSize);
            MPI_Irecv(recvBuffers[t][d].data(),
                      bufferSize * sizeof(Grid),
                      MPI_BYTE,
                      nbrRank, tag,
                      MPI_COMM_WORLD, &requests[reqCount]);
            reqCount++; // Increment the request count
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Ensure all receives are posted

    // Step 5B: Post all non-blocking sends using the packSendBuffer function
    for (int t = 0; t < numTiles; t++) {
        Tile &tile = info.tiles[t]; // Current tile
        for (int d = 0; d < 8; d++) { // Iterate over all 8 directions
            int tagSend = computeTag(rank, t, d, tilesPerRank); 
            vector<Grid> sendBuffer = packSendBuffer(tile, d, guard); // Pack the send buffer
            MPI_Isend(sendBuffer.data(),
                      sendBuffer.size() * sizeof(Grid),
                      MPI_BYTE,
                      tile.neighborRank[d], tagSend,
                      MPI_COMM_WORLD, &requests[reqCount]);
            reqCount++; // Increment the request count
        }
    }

    // Wait for all communication to complete
    MPI_Waitall(reqCount, requests.data(), MPI_STATUSES_IGNORE);

    // Step 6: Update guard regions using the updateGuardRegion function
    for (int t = 0; t < numTiles; t++) {
        Tile &tile = info.tiles[t]; // Current tile
        for (int d = 0; d < 8; d++) { // Iterate over all 8 directions
            updateGuardRegion(tile, d, guard, recvBuffers[t][d]);
        }
    }

    // Prints for testing
    MPI_Barrier(MPI_COMM_WORLD);
    for (int p = 0; p < size; p++) { 
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            cout << "\n===== OUTPUT FOR RANK " << rank << " (Coordinates: " 
            << info.rankRow << ", " << info.rankCol << ") =====\n";       
            for (int t = 0; t < numTiles; t++) {
                Tile &tile = info.tiles[t];
                int totalX = tile.nx + 2 * guard;
                int totalY = tile.ny + 2 * guard;
                cout << "Tile[" << tile.tileRow << "," << tile.tileCol << "]\n";
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
                // Oblique (corner) regions
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
