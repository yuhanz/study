package com.edmunds.crossshop.consumer;

import com.edmunds.crossshop.SlaPolicies;
import com.edmunds.crossshop.repository.CrossShopRedisRepository;
import com.edmunds.dwh.drt.userevent.gen.CrossShopMetrics;
import com.edmunds.dwh.drt.userevent.gen.CrossShopMetricsService;
import com.edmunds.eps.endpoint.Receiver;
import com.edmunds.eps.handlers.AbstractDataHandler;
import com.edmunds.monitoring.Monitoring;
import org.apache.thrift.TException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.jmx.export.annotation.ManagedResource;
import org.springframework.stereotype.Component;

import java.io.IOException;

import static com.edmunds.monitoring.Monitoring.debug;
//import static com.edmunds.monitoring.Monitoring.violateSla;

/**
 * @author yzhang
 */
@Component
@ManagedResource(objectName = "Edmunds:type=EPS,concern=Consumer,name=CrossShopRedisConsumer")
public class CrossShopRedisConsumer extends AbstractDataHandler implements CrossShopMetricsService.Iface {


    private CrossShopRedisRepository repository;

    /**
     * Instantiates a new tco consumer.
     *
     * @param receiver              {@link Receiver}.
     */
    @Autowired
    public CrossShopRedisConsumer(Receiver receiver, CrossShopRedisRepository repository) {
        super(receiver);
        this.repository = repository;
    }

    @Override
    public void updateCrossShopMetrics(CrossShopMetrics crossShopMetrics) throws TException {
        try {
            repository.save(crossShopMetrics);
        } catch (IOException e) {
            new TException(e);
        }
    }

    /**
     * Main method.
     *
     * @param args the arguments.
     */
    public static void main(String[] args) {
        try (ClassPathXmlApplicationContext context = new ClassPathXmlApplicationContext(
            "META-INF/spring/cross-shop-redis-consumer.xml")) {
            debug(context, "CrossShop Redis Consumer application context loaded: " + context);
            try {
                synchronized(CrossShopRedisConsumer.class) {
                    CrossShopRedisConsumer.class.wait();
                }
            } catch(InterruptedException e) {
                Monitoring.violateSla(SlaPolicies.CRITICAL_FAILED.value(), e);
            }
        }
    }


}
