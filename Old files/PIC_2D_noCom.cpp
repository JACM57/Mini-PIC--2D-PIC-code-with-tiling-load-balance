#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

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
    // Vector of particles
    vector<Particle> particles;
    // 1D vector of grid cells, ordered in row-major format
    vector<Grid> grid;
    // Position in the local tile grid
    int tileRow, tileCol;
    // Info about its eight neighbors
    int neighborRank[8];
    int neighborTile[8];
};

// RankInfo structure: track ranks and tiles in each rank
struct RankInfo {
    // Position in MPI grid
    int rank, rankRow, rankCol;
    // 2D tile layout inside each rank
    int tileRows, tileCols;
    // Vector of all tiles managed by this rank
    vector<Tile> tiles;
};

// Determine a near-square decomposition of 'size'
void findBestGrid(int size, int &R, int &C) {
    // Start with the integer square root of the total processes for rows (R) and adjusts R until it divides size evenly
    R = static_cast<int>( ::sqrt((double)size) );
    while (R > 1 && (size % R != 0)) R--;
    // The number of columns (C) is then calculated as size / R
    C = size / R;
}

// Compute a neighbor tile with periodic boundary conditions
pair<int,int> getNeighbor2D(
    // Current tile position within a rank
    int tRow, int tCol,
    // Rank position in the MPI grid
    int rankRow, int rankCol,
    // Local grid dimensions
    int tileRows, int tileCols,
    // Global MPI grid dimensions (R rows and C columns)
    int R, int C,
    // Displacement -> specify the direction of the neighbor relative to the current tile
    // Example: dRow = 0, dCol = -1 -> neighbor is to the left
    int dRow, int dCol
) {
    int newTrow = tRow + dRow;
    int newTcol = tCol + dCol;
    // By default assume that the neighbor is on the same MPI rank (adjust if it crosses rank boundaries)
    int newRankRow = rankRow;
    int newRankCol = rankCol;

    if (newTrow < 0) { // The neighbor is above the top of the tile grid
        newTrow = tileRows - 1; // Wrap around to the bottom
        newRankRow = (rankRow == 0) ? R - 1 : rankRow - 1;
    } else if (newTrow >= tileRows) { // The neighbor is below the bottom of the tile grid
        newTrow = 0; // Wrap around to the top
        newRankRow = (rankRow == R - 1) ? 0 : rankRow + 1;
    }
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

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI ranks are organized in a grid with R rows and C columns
    int R, C;
    findBestGrid(size, R, C);
    // Use integer division so that each row contains C ranks
    // Example: if C=3, ranks 0,1,2 are in row 0, ranks 3,4,5 are in row 1, ...
    int rankRow = rank / C;
    int rankCol = rank % C;

    // Example: 3x3 tiles in each rank => 9 tiles total
    int tileRows = 3;
    int tileCols = 3;
    int numTiles = tileRows * tileCols;

    // Store information about the current rank and its tiles
    RankInfo info;
    info.rank = rank;
    info.rankRow = rankRow;
    info.rankCol = rankCol;
    info.tileRows = tileRows;
    info.tileCols = tileCols;
    info.tiles.resize(numTiles);

    for(int t=0; t<numTiles; t++){
        info.tiles[t].tileRow = t / tileCols;
        info.tiles[t].tileCol = t % tileCols;
    }

    // Compute neighbors for each tile
    // Order: left, right, up, down, top-left, top-right, bottom-left, bottom-right
    static const int dRow[8] = { 0,  0, -1,  1, -1, -1,  1,  1 };
    static const int dCol[8] = {-1,  1,  0,  0, -1,  1, -1,  1 };

    for(int t=0; t<numTiles; t++){
        int tRow = info.tiles[t].tileRow; // Tile row within the rank
        int tCol = info.tiles[t].tileCol; // Tile column within the rank
        for(int n=0; n<8; n++){
            pair<int,int> nb = getNeighbor2D( // Get neighbor (nb) rank and tile index
                tRow, tCol, rankRow, rankCol,
                tileRows, tileCols, R, C,
                dRow[n], dCol[n]
            );
            info.tiles[t].neighborRank[n] = nb.first; // Store neighbor rank
            info.tiles[t].neighborTile[n] = nb.second; // Store neighbor tile index
        }
    }

    // Print tile and neighbor information
    MPI_Barrier(MPI_COMM_WORLD);
    for(int p=0; p<size; p++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == p){
            cout << "\nRank " << rank << " (Row=" << rankRow << ",Col=" << rankCol
                 << ") => tileRows=" << tileRows << ", tileCols=" << tileCols << endl;
            for(int t=0; t<numTiles; t++){
                cout << "  Tile[" << info.tiles[t].tileRow << "," << info.tiles[t].tileCol << "]\n";
                static const char* dirs[8] = {
                    "Left", "Right", "Up", "Down","TopLeft","TopRight","BottomLeft","BottomRight"
                };
                for(int n=0; n<8; n++){
                    cout << "    " << dirs[n] << ": rank="
                         << info.tiles[t].neighborRank[n]
                         << ", tile=" << info.tiles[t].neighborTile[n] << "\n";
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
