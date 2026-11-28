# Shared acceleration projection — 2026-10-22

Baseline: `19e0c7a`. Core adaptaccel projects traversal arrays, procedural data
and instance records, drops the cyclic native owner and retains the miss sentinel.
Both backend copies are deleted. Vulkan passes identity for scene conversion;
Metal passes Adapt.adapt bound to its adaptor. The callback runs in constructor
field order, preserving adaptation/error order rather than converting scene early.

126 host/AMD/NVIDIA assertions pass: geometry preparation, explicit scene policies
and conversion order with an effectful host adaptor, batched triangle lookup,
instance synchronization and procedural tracing. Six paired checks preserve
native projected field types/values and allocation. Forcing 100 returned values
into Ref{Any} allocates 8,000 bytes in both arms; no allocation gain is claimed.
Validation and synchronization validation drains are empty.

All 115 source files parse; prose edits preserve definition trees. Source falls
50,892 → 50,824 physical and 22,889 → 22,877 implementation lines (12 removed).
Metal's existing scene conversion is preserved in source and exercised by the
host projection contract; native Metal execution remains unqualified here.
