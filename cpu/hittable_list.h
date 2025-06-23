#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "rtweekend.h"

#include <unordered_map>
#include <string>

class hittable_map : public hittable {
public:
    std::unordered_map<std::string, shared_ptr<hittable>> objects;

    hittable_map() {}
    hittable_map(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object) {
        // Assume que hittable tem um método id() que retorna std::string
        objects[object->id()] = object;
    }

    void remove(const std::string& id) {
        objects.erase(id);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto& [name, object] : objects) {
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif // !HITTABLE_LIST_H

