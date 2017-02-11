package com.devon.demo.java.model;

public class Person {

//	public static void main(String[] args) {
//	}

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    @Override
    public String toString() {
        return firstName +" "+ lastName;
    }

    private String firstName;
    private String lastName;

}
