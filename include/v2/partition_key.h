#pragma once

#include <cstdint>
#include <functional>

namespace diskann {

    using LabelId  = uint32_t;
    using TenantId = uint32_t;

    struct PartitionKey {
        LabelId  label  = 0;
        TenantId tenant = 0;

        bool operator==(const PartitionKey& o) const {
            return label == o.label && tenant == o.tenant;
        }
    };

    struct PartitionKeyHash {
        size_t operator()(const PartitionKey& k) const {
            return (static_cast<size_t>(k.label) << 32) ^
                   static_cast<size_t>(k.tenant);
        }
    };

}