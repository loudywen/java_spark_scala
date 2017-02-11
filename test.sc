

class Person( @scala.beans.BeanProperty var name:String = "",  @scala.beans.BeanProperty var age:Int){
  name = name.toUpperCase

  override def toString: String = "name: "+name + " age: "+age
}
