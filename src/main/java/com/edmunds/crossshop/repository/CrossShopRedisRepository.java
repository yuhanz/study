package com.edmunds.crossshop.repository;

import com.edmunds.crossshop.SlaPolicies;
import com.edmunds.dwh.drt.userevent.gen.CrossShopMetric;
import com.edmunds.dwh.drt.userevent.gen.CrossShopMetrics;
import com.edmunds.dwh.drt.userevent.gen.DMACrossShopMetric;
import com.edmunds.monitoring.Monitoring;
import org.codehaus.jackson.map.ObjectMapper;
import org.springframework.stereotype.Component;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.Protocol;
import rx.Observable;
import rx.Subscriber;
import rx.functions.Action1;
import rx.functions.Func1;

import java.io.IOException;
import java.util.Set;

/**
 * Created by yzhang on 11/28/16.
 */
@Component
public class CrossShopRedisRepository {
    private String host = null;
    private int port = 6379;
    private int timeout = Protocol.DEFAULT_TIMEOUT;

    private JedisPool jedisPool;
    private ObjectMapper itemMapper = new ObjectMapper();

    public CrossShopRedisRepository() {
        jedisPool = new JedisPool(new JedisPoolConfig(), host, port, timeout);
    }

    public void save(CrossShopMetrics crossShopMetrics) throws IOException {

        Observable<CrossShopMetricDocument> metrics = convertCrossShopMetrics(crossShopMetrics);

        Observable<CrossShopMetricDocument> dmaMetrics = Observable.from(crossShopMetrics.getDateDMACrossShopMetrics().values())
                .flatMap(new Func1<Set<DMACrossShopMetric>, Observable<CrossShopMetricDocument>>() {
                    @Override
                    public Observable<CrossShopMetricDocument> call(final Set<DMACrossShopMetric> crossShopMetrics) {
                        return Observable.create(new Observable.OnSubscribe<CrossShopMetricDocument>() {
                            @Override
                            public void call(Subscriber<? super CrossShopMetricDocument> subscriber) {
                                for (DMACrossShopMetric dmaMetric : crossShopMetrics) {
                                    String dmaCode = dmaMetric.getDmaCode();
                                    for (CrossShopMetric metric : dmaMetric.getCrossShopMetrics()) {
                                        subscriber.onNext(new CrossShopMetricDocument(metric, dmaCode));
                                    }
                                }
                                subscriber.onCompleted();
                            }
                        });
                    }
                });

        Observable.concat(metrics, dmaMetrics).subscribe(new Action1<CrossShopMetricDocument>() {
            @Override
            public void call(CrossShopMetricDocument document) {
                try {
                    save(document);
                } catch (IOException e) {
                    Monitoring.violateSla(SlaPolicies.REDIS_WRITE_FAILED.value(), e);
                    throw new RuntimeException(e);
                }
            }
        });
    }

    private void save(CrossShopMetricDocument document) throws IOException {
        try {
            String content = itemMapper.writeValueAsString(document);
            Jedis jedis = jedisPool.getResource();
            jedis.set(document.getModelLinkCode(), content);
        } catch (IOException e) {
            Monitoring.violateSla(SlaPolicies.REDIS_WRITE_FAILED.value(), e);
            throw e;
        }

    }

    private Observable<CrossShopMetricDocument> convertCrossShopMetrics(CrossShopMetrics crossShopMetrics) {
        return Observable.from(crossShopMetrics.getDateCrossShopMetrics().values())
                .flatMap(new Func1<Set<CrossShopMetric>, Observable<CrossShopMetricDocument>>() {
                    @Override
                    public Observable<CrossShopMetricDocument> call(final Set<CrossShopMetric> crossShopMetrics) {
                        return Observable.create(new Observable.OnSubscribe<CrossShopMetricDocument>() {
                            @Override
                            public void call(Subscriber<? super CrossShopMetricDocument> subscriber) {
                                for (CrossShopMetric metric : crossShopMetrics) {
                                    subscriber.onNext(new CrossShopMetricDocument(metric));
                                }
                                subscriber.onCompleted();
                            }
                        });
                    }
                });
    }

    public void setHost(String host) {
        this.host = host;
    }

    public void setPort(int port) {
        this.port = port;
    }

    public void setTimeout(int timeout) {
        this.timeout = timeout;
    }
}


