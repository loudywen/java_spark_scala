package com.devon.demo.java.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by diwenlao on 2/8/17.
 */
public class JavaUtility {

    private static Logger logger = LoggerFactory.getLogger(JavaUtility.class);

    /**
     * Gets data file.
     *
     * @param fileName the file name
     * @return the data file
     */
    public static File getDataFile(String fileName) {
        File inputFile = null;
        try {
//            URL data = JavaUtility.class.getResource("/"+fileName);
//            URL data = new URL(fileName);

            inputFile = new File(fileName);
            logger.info("trying with file  = " + inputFile.getAbsolutePath());

            if(!inputFile.exists()) {
                logger.info("trying over with file  = " + inputFile.getAbsolutePath());
                throw new RuntimeException("input file not found: " + inputFile.getAbsolutePath());
            }
        } catch (Exception e) {
            // no data found
            logger.error(e.getMessage());
        }
        return inputFile;
    }

}
