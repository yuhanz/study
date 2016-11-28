package com.edmunds.crossshop;

import com.edmunds.monitoring.api.sla.PolicyLocation;

/**
 * Created by yzhang on 11/28/16.
 */
public enum SlaPolicies {
    CRITICAL_FAILED(PolicyLocation.valueOf("cross-shop-redis-consumer:critical-failed")),
    REDIS_WRITE_FAILED(PolicyLocation.valueOf("cross-shop-redis-consumer:redis-write-failed"));

    private final PolicyLocation policyLocation;

    SlaPolicies(PolicyLocation policyLocation) {
        this.policyLocation = policyLocation;
    }

    public PolicyLocation value() {
        return policyLocation;
    }

}