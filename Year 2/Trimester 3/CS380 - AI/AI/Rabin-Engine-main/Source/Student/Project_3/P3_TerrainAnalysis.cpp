#include <pch.h>
#include "Terrain/TerrainAnalysis.h"
#include "Terrain/MapMath.h"
#include "Agent/AStarAgent.h"
#include "Terrain/MapLayer.h"
#include "Projects/ProjectThree.h"

#include <iostream>
#define CAST(t, v) (static_cast<t>(v))

namespace
{
    bool Equal(float a, float b)
    {
        return fabs(a - b) < FLT_EPSILON;
    }
}

bool ProjectThree::implemented_fog_of_war() const // extra credit
{
    return false;
}

float distance_to_closest_wall(int row, int col)
{
    /*
        Check the euclidean distance from the given cell to every other wall cell,
        with cells outside the map bounds treated as walls, and return the smallest
        distance.  Make use of the is_valid_grid_position and is_wall member
        functions in the global terrain to determine if a cell is within map bounds
        and a wall, respectively.
    */
    float closestWallDist{ FLT_MAX };
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };

    auto IsWall = [](int r, int c)
    {
        return !terrain->is_valid_grid_position(r, c) || terrain->is_wall(r, c);
    };

    for (int r{ -1 }; r <= HEIGHT; ++r)
    {
        for (int c{ -1 }; c <= WIDTH; ++c)
        {
            if (!IsWall(r, c))
                continue;
            float const dx = static_cast<float>(col - c),
                        dy = static_cast<float>(row - r);
            float const dist = dx * dx + dy * dy;
            if (dist < closestWallDist)
                closestWallDist = dist;
        }
    }

    return sqrt(closestWallDist);
}

bool is_clear_path(int row0, int col0, int row1, int col1)
{
    /*
        Two cells (row0, col0) and (row1, col1) are visible to each other if a line
        between their centerpoints doesn't intersect the four boundary lines of every
        wall cell.  You should puff out the four boundary lines by a very tiny amount
        so that a diagonal line passing by the corner will intersect it.  Make use of the
        line_intersect helper function for the intersection test and the is_wall member
        function in the global terrain to determine if a cell is a wall or not.
    */
    if (!terrain->is_valid_grid_position(row0, col0) || !terrain->is_valid_grid_position(row1, col1))
        return false;
    struct Line
    {
        Vec2 p0{}, p1{};
        // los -> Line of sight
    } const los{ { CAST(float, col0), CAST(float, row0) },
                 { CAST(float, col1), CAST(float, row1) } };

    // WRITE YOUR CODE HERE
    int const MIN_ROW = std::min(row0, row1),
              MAX_ROW = std::max(row0, row1);
    int const MIN_COL = std::min(col0, col1),
              MAX_COL = std::max(col0, col1);

    bool pathClear{ true };

    auto Intersect = [](Line const& l0, Line const& l1)
    {
        return line_intersect(l0.p0, l0.p1, l1.p0, l1.p1);
    };

	for (int r{ MIN_ROW }; r <= MAX_ROW; ++r)
	{
		for (int c{ MIN_COL }; c <= MAX_COL; ++c)
        {
            if (!terrain->is_wall(r, c))
                continue;

            Line const top  { { CAST(float, c) + 0.501f, CAST(float, r) + 0.501f },
                              { CAST(float, c) - 0.501f, CAST(float, r) + 0.501f } };
            Line const btm  { { CAST(float, c) + 0.501f, CAST(float, r) - 0.501f },
                              { CAST(float, c) - 0.501f, CAST(float, r) - 0.501f } };
            Line const right{ { CAST(float, c) + 0.501f, CAST(float, r) + 0.501f },
                              { CAST(float, c) + 0.501f, CAST(float, r) - 0.501f } };
            Line const left { { CAST(float, c) - 0.501f, CAST(float, r) + 0.501f },
                              { CAST(float, c) - 0.501f, CAST(float, r) - 0.501f } };

            // If any of the wall lines intersect with the los, path is not clear
            if (Intersect(los, top) || Intersect(los, btm) || Intersect(los, right) || Intersect(los, left))
            {
                pathClear = false;
                break;
            }
        }
        if (!pathClear)
            break;
    }

    return pathClear; // REPLACE THIS
}

void analyze_openness(MapLayer<float> &layer)
{
    /*
        Mark every cell in the given layer with the value 1 / (d * d),
        where d is the distance to the closest wall or edge.  Make use of the
        distance_to_closest_wall helper function.  Walls should not be marked.
    */

    // WRITE YOUR CODE HERE
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };
    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            if (terrain->is_wall(r, c))
                continue;
            float const d = distance_to_closest_wall(r, c);
            layer.set_value(r, c, 1.0f / (d * d));
        }
    }
}

void analyze_visibility(MapLayer<float> &layer)
{
    /*
        Mark every cell in the given layer with the number of cells that
        are visible to it, divided by 160 (a magic number that looks good).  Make sure
        to cap the value at 1.0 as well.

        Two cells are visible to each other if a line between their centerpoints doesn't
        intersect the four boundary lines of every wall cell.  Make use of the is_clear_path
        helper function.
    */

    // WRITE YOUR CODE HERE
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };
    int const MAX_CELLS{ WIDTH * HEIGHT };
    for (int i{}; i < MAX_CELLS; ++i)
    {
        int numOfVisibleCells{};
        int const r0 = i / HEIGHT,
                  c0 = i % WIDTH;
        if (terrain->is_wall(r0, c0))
            continue;
        for (int j{}; j < MAX_CELLS; ++j)
        {
            int const r1 = j / HEIGHT,
                      c1 = j % WIDTH;
            if (is_clear_path(r0, c0, r1, c1))
                ++numOfVisibleCells;
        }
        float const val = std::clamp(CAST(float, numOfVisibleCells / 160.0f), 0.0f, 1.0f);
        layer.set_value(r0, c0, val);
    }
}

void analyze_visible_to_cell(MapLayer<float> &layer, int row, int col)
{
    /*
        For every cell in the given layer mark it with 1.0
        if it is visible to the given cell, 0.5 if it isn't visible but is next to a visible cell,
        or 0.0 otherwise.

        Two cells are visible to each other if a line between their centerpoints doesn't
        intersect the four boundary lines of every wall cell.  Make use of the is_clear_path
        helper function.
    */
    // WRITE YOUR CODE HERE
    struct Coords
    {
        int r{}, c{};
    };
    layer.set_value(row, col, 1.0f);
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };
    std::vector<Coords> coords{}; coords.reserve(CAST(size_t, WIDTH)* CAST(size_t, HEIGHT));

    // Loop to assign map layer with values 1.0f/0.0f
    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            if ( terrain->is_wall(r, c) || (r == row && c == col) )
                continue;
            if (is_clear_path(row, col, r, c))
            {
                layer.set_value(r, c, 1.0f);
                coords.emplace_back(Coords{ r, c });
            }
            else
                layer.set_value(r, c, 0.0f);
        }
    }

    size_t index = 0;
    auto IsValid = [&layer, &coords, &index](int r, int c)
    {
        return terrain->is_valid_grid_position(r, c) 
            && !terrain->is_wall(r, c) 
            && layer.get_value(r, c) != 1.0f
            && is_clear_path(r, c, coords[index].r, coords[index].c);
    };

    do  // loop through each visible cell to search for valid visible neighbours (to set each cells to be 0.5f)
    {
        int const r = coords[index].r,
                  c = coords[index].c;
        // Top Left
        if (IsValid(r + 1, c - 1))
            layer.set_value(r + 1, c - 1, 0.5f);
        // Top
        if(IsValid(r + 1, c))
            layer.set_value(r + 1, c, 0.5f);
        // Top Right
        if (IsValid(r + 1, c + 1))
            layer.set_value(r + 1, c + 1, 0.5f);
        // Left
        if (IsValid(r, c - 1))
            layer.set_value(r, c - 1, 0.5f);
        // Right
        if (IsValid(r, c + 1))
            layer.set_value(r, c + 1, 0.5f);
        // Btm left
        if (IsValid(r - 1, c - 1))
            layer.set_value(r - 1, c - 1, 0.5f);
        // Btm
        if (IsValid(r - 1, c))
            layer.set_value(r - 1, c, 0.5f);
        // Btm right
        if (IsValid(r - 1, c + 1))
            layer.set_value(r - 1, c + 1, 0.5f);
    } while (++index < coords.size());
}

void analyze_agent_vision(MapLayer<float> &layer, const Agent *agent)
{
    /*
        For every cell in the given layer that is visible to the given agent,
        mark it as 1.0, otherwise don't change the cell's current value.

        You must consider the direction the agent is facing.  All of the agent data is
        in three dimensions, but to simplify you should operate in two dimensions, the XZ plane.

        Take the dot product between the view vector and the vector from the agent to the cell,
        both normalized, and compare the cosines directly instead of taking the arccosine to
        avoid introducing floating-point inaccuracy (larger cosine means smaller angle).

        Give the agent a field of view slighter larger than 180 degrees.

        Two cells are visible to each other if a line between their centerpoints doesn't
        intersect the four boundary lines of every wall cell.  Make use of the is_clear_path
        helper function.
    */

    // WRITE YOUR CODE HERE
    GridPos const& pos{ terrain->get_grid_position( agent->get_position() ) };
    Vec2 const& view{ agent->get_forward_vector().x, agent->get_forward_vector().z };
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };

    // ANGLE is cos95
    float constexpr const ANGLE = -0.087f;

    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            if (!is_clear_path(pos.row, pos.col, r, c))
                continue;

            // Get the dot product with agent's view vector and vector from agent to cell
            Vec2 v{ terrain->get_world_position(r, c).x - agent->get_position().x,
                    terrain->get_world_position(r, c).z - agent->get_position().z }; v.Normalize();
            if (v.Dot(view) < ANGLE)
                continue;
            layer.set_value(r, c, 1.0f);
        }
    }
}

void propagate_solo_occupancy(MapLayer<float> &layer, float decay, float growth)
{
    /*
        For every cell in the given layer:

            1) Get the value of each neighbor and apply decay factor
            2) Keep the highest value from step 1
            3) Linearly interpolate from the cell's current value to the value from step 2
               with the growing factor as a coefficient.  Make use of the lerp helper function.
            4) Store the value from step 3 in a temporary layer.
               A float[40][40] will suffice, no need to dynamically allocate or make a new MapLayer.

        After every cell has been processed into the temporary layer, write the temporary layer into
        the given layer;
    */
    
    // WRITE YOUR CODE HERE
    float tmp[40][40];  // [r][c]
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };

    auto IsValid = [](int r, int c)
    {
        return terrain->is_valid_grid_position(r, c) && !terrain->is_wall(r, c);
    };

    auto Decay = [decay](float influenceValue, float ex)
    {
        return influenceValue * exp(ex * decay);
    };

    auto Prop = [&layer, growth, &tmp, IsValid, Decay](int r, int c)
    {
        float max{ FLT_MIN };
        float vals[8]{ -FLT_MIN, -FLT_MIN, -FLT_MIN, -FLT_MIN, -FLT_MIN, -FLT_MIN, -FLT_MIN, -FLT_MIN };
        size_t index{};

        float constexpr const sqrt2{ 1.41421356237f };

        // Top
        if (IsValid(r + 1, c))
            vals[index++] = Decay(layer.get_value(r + 1, c), -1.0f);
        // Btm
        if (IsValid(r - 1, c))
            vals[index++] = Decay(layer.get_value(r - 1, c), -1.0f);
        // Right
        if (IsValid(r, c + 1))
            vals[index++] = Decay(layer.get_value(r, c + 1), -1.0f);
        // Left
        if (IsValid(r, c - 1))
            vals[index++] = Decay(layer.get_value(r, c - 1), -1.0f);
        
        // Top Left
        if ( IsValid(r + 1, c) && IsValid(r, c - 1) )
            vals[index++] = Decay(layer.get_value(r + 1, c - 1), -sqrt2);
        // Top Right
        if( IsValid(r + 1, c) && IsValid(r, c + 1) )
            vals[index++] = Decay(layer.get_value(r + 1, c + 1), -sqrt2);
        // Btm Left
        if( IsValid(r - 1, c) && IsValid(r, c - 1) )
            vals[index++] = Decay(layer.get_value(r - 1, c - 1), -sqrt2);
        // Btm Right
		if ( IsValid(r - 1, c) && IsValid(r, c + 1) )
            vals[index] = Decay(layer.get_value(r - 1, c + 1), -sqrt2);

        for (size_t i{}; i < 8; ++i)
        {
            if (vals[i] == -FLT_MIN)
                break;
            if (max < vals[i])
                max = vals[i];
        }

        return lerp(layer.get_value(r, c), max, growth);
    };

    // Propagate the cells
    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            if (terrain->is_wall(r, c))
                continue;
            tmp[r][c] = Prop(r, c);
        }
    }

    // assigning temp values into layer
    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            if (terrain->is_wall(r, c))
                continue;
            layer.set_value(r, c, tmp[r][c]);
        }
    }
}

void propagate_dual_occupancy(MapLayer<float> &layer, float decay, float growth)
{
    /*
        Similar to the solo version, but the values range from -1.0 to 1.0, instead of 0.0 to 1.0

        For every cell in the given layer:

        1) Get the value of each neighbor and apply decay factor
        2) Keep the highest ABSOLUTE value from step 1
        3) Linearly interpolate from the cell's current value to the value from step 2
           with the growing factor as a coefficient.  Make use of the lerp helper function.
        4) Store the value from step 3 in a temporary layer.
           A float[40][40] will suffice, no need to dynamically allocate or make a new MapLayer.

        After every cell has been processed into the temporary layer, write the temporary layer into
        the given layer;
    */

    // WRITE YOUR CODE HERE
}

void normalize_solo_occupancy(MapLayer<float> &layer)
{
    /*
        Determine the maximum value in the given layer, and then divide the value
        for every cell in the layer by that amount.  This will keep the values in the
        range of [0, 1].  Negative values should be left unmodified.
    */

    // WRITE YOUR CODE HERE
    float max{ -FLT_MIN };
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };
    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            float const val = layer.get_value(r, c);
            if (max < val)
                max = val;
        }
    }

    // Don't let value be divided by 0
    if (Equal(max, 0.0f)) 
        return;

    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            float const val = layer.get_value(r, c);
            if (terrain->is_wall(r, c) || val <= 0.0f)
                continue;
            layer.set_value(r, c, val / max);
        }
    }
}

void normalize_dual_occupancy(MapLayer<float> &layer)
{
    /*
        Similar to the solo version, but you need to track greatest positive value AND 
        the least (furthest from 0) negative value.

        For every cell in the given layer, if the value is currently positive divide it by the
        greatest positive value, or if the value is negative divide it by -1.0 * the least negative value
        (so that it remains a negative number).  This will keep the values in the range of [-1, 1].
    */

    // WRITE YOUR CODE HERE
}

void enemy_field_of_view(MapLayer<float> &layer, float fovAngle, float closeDistance, float occupancyValue, AStarAgent *enemy)
{
    /*
        First, clear out the old values in the map layer by setting any negative value to 0.
        Then, for every cell in the layer that is within the field of view cone, from the
        enemy agent, mark it with the occupancy value.  Take the dot product between the view
        vector and the vector from the agent to the cell, both normalized, and compare the
        cosines directly instead of taking the arccosine to avoid introducing floating-point
        inaccuracy (larger cosine means smaller angle).

        If the tile is close enough to the enemy (less than closeDistance),
        you only check if it's visible to enemy.  Make use of the is_clear_path
        helper function.  Otherwise, you must consider the direction the enemy is facing too.
        This creates a radius around the enemy that the player can be detected within, as well
        as a fov cone.
    */

    // WRITE YOUR CODE HERE
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };

    // Clear out old values by setting negative values to be 0.0f
    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            if (terrain->is_wall(r, c) || layer.get_value(r, c) >= 0.0f)
                continue;
            layer.set_value(r, c, 0.0f);
        }
    }

    // To draw the pov for enemy based on it's view vector
    float const ANGLE = cos( MyVar::DegToRad( fovAngle * 0.5f ) );
    Vec2 const& view{ enemy->get_forward_vector().x, enemy->get_forward_vector().z };
    GridPos const& pos{ terrain->get_grid_position( enemy->get_position() ) };
    float const SQUARE_DIST = closeDistance * closeDistance;

    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            if (!is_clear_path(pos.row, pos.col, r, c))
                continue;
            float const dx = pos.col - CAST(float, c),
                        dy = pos.row - CAST(float, r);
            if ((dx * dx + dy * dy) < SQUARE_DIST)
                layer.set_value(r, c, occupancyValue);

			// Get the dot product with agent's view vector and vector from agent to cell
			Vec2 v{ terrain->get_world_position(r, c).x - enemy->get_position().x,
					terrain->get_world_position(r, c).z - enemy->get_position().z }; v.Normalize();
			if (v.Dot(view) < ANGLE)
				continue;
			layer.set_value(r, c, occupancyValue);
        }
    }
}

bool enemy_find_player(MapLayer<float> &layer, AStarAgent *enemy, Agent *player)
{
    /*
        Check if the player's current tile has a negative value, ie in the fov cone
        or within a detection radius.
    */

    const auto &playerWorldPos = player->get_position();

    const auto playerGridPos = terrain->get_grid_position(playerWorldPos);

    // verify a valid position was returned
    if (terrain->is_valid_grid_position(playerGridPos) == true)
    {
        if (layer.get_value(playerGridPos) < 0.0f)
        {
            return true;
        }
    }

    // player isn't in the detection radius or fov cone, OR somehow off the map
    return false;
}

bool enemy_seek_player(MapLayer<float> &layer, AStarAgent *enemy)
{
    /*
        Attempt to find a cell with the highest nonzero value (normalization may
        not produce exactly 1.0 due to floating point error), and then set it as
        the new target, using enemy->path_to.

        If there are multiple cells with the same highest value, then pick the
        cell closest to the enemy.

        Return whether a target cell was found.
    */

    // WRITE YOUR CODE HERE
    int const WIDTH{ terrain->get_map_width() }, HEIGHT{ terrain->get_map_height() };
    float max{ -FLT_MIN };
    
    // First loop iteration: Find tiles with highest occupancy value
    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            float const val = layer.get_value(r, c);
            if (val < 0.0f)
                continue;

            if ( max < val )
                max = val;
        }
    }

    if (Equal(max, -FLT_MIN))
        return false;

    // Second loop iteration: Find the closest tile with the highest occupancy value
    Vec3 targetPos{};
    float nearest{ 5000.0f };   // because the maximum distance that can be achieve by 40x40 grid is 3200
    GridPos const& pos{ terrain->get_grid_position( enemy->get_position() ) };
    for (int r{}; r < HEIGHT; ++r)
    {
        for (int c{}; c < WIDTH; ++c)
        {
            float const dx = pos.col - CAST(float, c),
                        dy = pos.row - CAST(float, r);
            float const dist = dx * dx + dy * dy;
            if (layer.get_value(r, c) < max || dist > nearest * nearest)
                continue;
			nearest = dist;
            targetPos = terrain->get_world_position(r, c);
        }
    }

    enemy->path_to(targetPos);

    return true; // REPLACE THIS
}
