"""
    AABB(min::Point3f, max::Point3f)

A 3D axis-aligned bounding box, stored as two corner points. Used as the
input element type for `build_blas_aabb`; every backend reads boxes in place,
24 bytes apart, the layout of Vulkan's `VkAabbPositionsKHR` and Metal's
`MTLAxisAlignedBoundingBox`.
"""
struct AABB
    min::Point3f
    max::Point3f
end
