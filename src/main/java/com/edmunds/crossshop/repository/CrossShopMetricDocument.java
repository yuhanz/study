package com.edmunds.crossshop.repository;

import com.edmunds.dwh.drt.userevent.gen.CrossShopMetric;

/**
 * Created by yzhang on 11/28/16.
 */
class CrossShopMetricDocument {
    private String modelLinkCode;
    private String make;
    private String model;
    private String competitorMake;
    private String competitorModel;
    private double crossShop;
    private double reverseCrossShop;

    private String dmaCode;

    public CrossShopMetricDocument(CrossShopMetric metric) {
        this(metric, null);
    }

    public CrossShopMetricDocument(CrossShopMetric metric, String dmaCode) {
        //this.modelLinkCode = metric.modelLinkCode;
        this.make = metric.make;
        this.model = metric.model;
        this.competitorMake = metric.competitorMake;
        this.competitorModel = metric.competitorModel;
        this.crossShop = metric.crossShop;
        this.reverseCrossShop = metric.reverseCrossShop;

        this.dmaCode = dmaCode;
    }

    public double getCrossShop() {
        return crossShop;
    }

    public String getModelLinkCode() {
        return modelLinkCode;
    }

    public String getMake() {
        return make;
    }

    public String getModel() {
        return model;
    }

    public String getCompetitorMake() {
        return competitorMake;
    }

    public String getCompetitorModel() {
        return competitorModel;
    }

    public double getReverseCrossShop() {
        return reverseCrossShop;
    }

    public String getDmaCode() {
        return dmaCode;
    }
}
