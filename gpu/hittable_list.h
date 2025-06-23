#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include <vector>
#include <memory> // For shared_ptr

// A host-side container for hittable objects.
// This class is used on the CPU to build the scene. It does not perform any ray-object intersections itself.
// The objects it holds are converted to GPU-compatible structs by the renderer.
class hittable_list : public hittable
{
public:
    std::vector<shared_ptr<hittable>> objects;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object)
    {
        objects.push_back(object);
    }
};

#endif // !HITTABLE_LIST_H
