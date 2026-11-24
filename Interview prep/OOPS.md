# Initializer list
```cpp
class test{
	public:
	test(){
		cout<<"Default"<<endl;
	}
	test(int x){
		cout<<"Parameterized"<<endl;
	}
}

class Main{
	test t;
	public:
	Main():t(10){
	// This gives only parametirized initialization (Only once)
	}
	Main(){
		t = test(10)
		// First initialized with default values then changed to parameterized values;
	}
}
t = test(10);

```

**When is the initializer list mandatory (not just an optimization)?**
- `const` data members (can only be initialized, never assigned)
- reference members (`int& r;` — must be bound at construction)
- members whose class has **no** default constructor
- a base class with no default constructor (see the Inheritance "Examples" section below)

So `:t(10)` isn't only faster than `t = test(10)` (one construction vs default-construct-then-assign) — for the cases above it's the **only** legal way.

# Copy constructors

```cpp
// Not copy constructor  
test t1(10);       // (note: `test t1 = new test(10)` is wrong -
                   //  `new` returns test*, not a test object)
test t2; //First default constructor
t2 = t1; //Then copy-ASSIGNMENT operator (operator=), not the copy ctor

// Copy constructor
test t1(10);
test t2(t1);       // Copy constructor call
test t3 = t1;      // ALSO the copy constructor (initialization, not assignment)


```

**Rule of thumb:** copy *constructor* runs when a new object is being **created** from an existing one. Copy *assignment* (`operator=`) runs when an **already-constructed** object is overwritten.

**Copy constructor is created automatically by compiler**
This copy constructor simply does **one by one assignment**  => Pointers are still the same (Shallow copy)

If we need DEEP COPY we need custom copy constructors

I don't understand why we need to pass the object only as a reference and not as a value - passing value causes something like infinite loop of initializations? Why and How?

**Answer — yes, pass-by-value would cause infinite recursion:**
The copy constructor's signature is `test(const test& other)`.
Imagine it were `test(test other)` (by value) instead. To create the parameter `other`, the compiler must make a **copy** of the argument — and copying an object means *calling the copy constructor*. That call again needs to copy *its* argument by value → calls the copy constructor again → … infinite recursion. The compiler actually rejects this.
Passing by **reference** means `other` is just an alias for the caller's object — **no copy is made to bind the parameter**, so the recursion never starts.
Why `const`? So it can also bind to temporaries / `const` objects, and it documents that the source won't be modified while being copied.

```cpp
test(const test& other){      // correct
    this->x = other.x;        // deep-copy pointers here if any
}
```


# Destructors

```cpp
test t1;
test t2;
//Construct t1
//Construct t2
//Destruct t2
//Destruct t1
```

Reverse order of their creation

Dynamic allocation in class -> Need to write custom destructor

# this

```cpp

public:
test& setx(int x){        // return test& (reference), NOT test* -
	this->x = x;          // we return *this, which is an object, not a pointer
	return *this;
}
test& sety(int x){
	this->y = x;
	return *this;
}
p1.setx(5).sety(10);      // method chaining:
                          // setx(5) returns a reference to p1, so .sety(10)
                          // is then called on that same p1. Only works because
                          // each setter returns *this.


// Eg. the same chaining idea is why cout works:
cout << x << ' ' << y;    // operator<< returns the stream (ostream&),
                          // so the next << chains onto the same cout
```

Can't change "this"  (it's a `test* const` - the pointer itself is constant)

# Static members

Common integers to all the objects of the class

```cpp

class test{
public:
static int count;

test(){
	count++;
}

~test(){
	count--;
}

static int getCount(){
	return count;
}

}

// Better definition is outside the class
int test::count=0;

```

# Static member functions

-> Modify and access static data members
-> Static members don't have any "this" pointer
-> Static member functions can only access the static data types in the class

Called like
	test::getCount()


# Inheritance

Public , Protected, Private

Private -> Not accessible at all outside the class
Protected -> Not accessible outside class except the classes which are derived from this
Public -> Accessible to all

```cpp

class Person{
private:
	int id;
protected:
	string name;
	int number;
public:
	int age;
}

class student: public Person{  // protected remain protected, public remain public from the base class
class student: private Person{  // protected,public become private
class student: protected Person{  // protected,public become protected 
private:
...
public:
...
void print(){
cout<<id; //Error (NOT ACCESSIBLe)
cout<<name; //Accessible
cout<<age; //Accessible
}

}


```

# Virtual functions

Can create an object of derived type and use it using pointer of base type
```cpp

base{
public:
void meth(){
	print();
}
}

derived: public base{
void meth(){
	print();
}
}

base *x = new derived();
derived d;
base &b = d;

b.meth();  // base meth    (b is a reference -> use ., not ->)
d.meth();  // derived meth
x->meth(); // base meth    (x is a pointer -> use ->)

// But if meth is virtual
base{
public:
virtual void meth(){
	print();
}
}
// Then b.meth() ALSO calls derived meth (the base reference now
// dispatches on the real object type). Without virtual it called base meth.

// So virtual is what makes b (a base ref/ptr to a derived object)
// pick the derived version at runtime.
```

Usually used to make array of just one type for all the objects instead of creating separate arrays for each kind

[Function overriding] This is runtime polymorphism -> Based on the type of object the method is called (Same signature for methods - f(int a), f(int a))
[Function overloading] Compile time polymorphism -> Function signature is different f(int a,int b), f(int a)

But even for function overriding we can decide at compile time based on the object type which function should be called. But why is it not compile time polymorphism and it is runtime?

**Answer — in general the compiler does NOT know the real (dynamic) type at compile time.**
A base pointer/reference can point to *any* derived object, and which one is often decided only at **runtime**:
```cpp
base* b;
if (userInput == 1) b = new derived1();
else                b = new derived2();
b->meth();   // which meth()? depends on userInput → only known at runtime
```
Because the call goes through a base pointer/reference, the language **must** defer the decision to runtime (via the vtable). That's what makes it *runtime* polymorphism.

Your intuition is right that in a *trivial* case like `derived d; base& b = d; b.meth();` the compiler can see the exact type. Modern compilers do exploit this — it's called **devirtualization** (an optimization) and they turn it into a direct call. But that's an optimization on top of semantics that are defined as runtime dispatch; the general case still needs the vtable.

Overloading, by contrast, is resolved purely from the **static (declared) types of the arguments**, which are always known at compile time → compile-time polymorphism.

# VPTR and VTable

C++ Standard doesn't say anything about implementation of how these work

Most compilers: 

VPTR maintained for every object
VPTR points to VTable
VTable which is there for every class
When you create an object of a class the compiler adds some code to our constructor this code sets VPTR to VTable of this class
VTable stores addresses of all virtual functions 

**Explained step by step:**

1. **VTable (one per class, built at compile time):** an array of function pointers, one slot per virtual function. For `base`, slot 0 → `&base::meth`. For `derived` (which overrides `meth`), its own vtable's slot 0 → `&derived::meth`. Same slot *index*, different address.

2. **VPTR (one hidden pointer per object):** the compiler secretly adds this pointer as the first member of every object of a class that has virtual functions. The constructor sets it to point to **that class's** vtable. So a `derived` object's VPTR points to `derived`'s vtable.

3. **A virtual call** like `b->meth()` is NOT compiled to "call `base::meth`". It's compiled to roughly:
   ```
   addr = b->vptr->vtable[slot_of_meth];   // look up at RUNTIME
   call addr;
   ```
   Because `b` points to a `derived` object, `b->vptr` points to `derived`'s vtable, so `slot_of_meth` holds `&derived::meth` → derived version runs. This is exactly *why* it's runtime polymorphism.

4. **Non-virtual call** skips all this — the address is hard-coded at compile time from the pointer's static type (`base`), so you'd get `base::meth`.

**One-line summary:** virtual call = "follow the object's VPTR → its class's VTable → the slot for this function → call whatever address is there." The object itself carries the information about which version to run.

```
 derived object            derived's VTable
 +----------+              +------------------+
 | vptr ----|------------->| [0] &derived::meth|
 | ...data  |              | [1] &derived::foo |
 +----------+              +------------------+
```

Override keyword is optional (Used for ease of debugging mostly)

# Examples
```cpp

class base{
protected:
	int x;
public:
	base(int a):x(a){
	
	}

};

class derived: public base{
private:
	int y;
public:
	derived(int b): y(b){
	}
}

int main(){
	derived d(10);
	d.print();
	return 0;
}
```

This code gives error because we are creating a derived class and base class should be created but we are not calling base class constructor and when compiler tries to call it we only made a constructor with one argument in function so it will try to call default one without arguments but it fails because there is no such constructor

```cpp

derived(int a,int b): base(a), y(b){

}


derived d(10,20);

// This will fix it
```

Creating an object calls base class constructor then derived class constructor
-> if we don't call base class constructor explicitly default base class constructor without parameters is run and if it is not there then there will be an error

# Multiple inheritance

B: C, A:B -> Multilevel

## Multiple 

A : B,C -> Multiple
Constructor B
Constructor C
Constructor A

### Problem
Not allowed by many problems -> Java doesn't allowed

Diamond problem
B: A, C: A
D: B, C

A has a member x then D obtains 2 x variables which is through two paths
A's constructor is called two times through two paths

### Solution
Use virtual keyword

B: virtual A, C: virtual A

# Operator overloading
```cpp

class test{
	
	test operator + (complex const &obj){
		test z;
		z.x = this.x + obj.x;
		z.y = this.x - obj.x;
		return z;
	}

}

```

Can't overload . , ::, ?: and sizeof 

**Member vs global — and when you're forced to use global:**

```cpp
// MEMBER function: left operand is "this", right operand is the argument
class complex {
    int x, y;
public:
    complex(int a=0, int b=0): x(a), y(b) {}
    complex operator+(const complex& obj) const {   // a + b -> a.operator+(b)
        return complex(x + obj.x, y + obj.y);
    }
};
```

```cpp
// GLOBAL (non-member) function: BOTH operands are explicit arguments
complex operator+(const complex& a, const complex& b) {
    return complex(a.x + b.x, a.y + b.y);   // needs access to x,y -> make it friend
}
```

**Why global is sometimes necessary:**
- If the **left** operand is not your class, a member function can't work.
  Example: `2 + obj`. A member `operator+` would need `2` to be a `complex` on the left (`(2).operator+`) — impossible. A global `operator+(int, complex)` (or a `complex(int)` converting constructor) handles it.
- `operator<<` and `operator>>` for streams **must** be global, because the left operand is `cout`/`cin` (an `ostream`/`istream`), not your class:
```cpp
ostream& operator<<(ostream& os, const complex& c) {
    os << c.x << " + " << c.y << "i";
    return os;            // return the stream so << can chain
}
// usage: cout << obj << endl;
```

**Assignment operator (`operator=`)** — must be a **member**, and should return `*this` by reference so `a = b = c` chains:
```cpp
complex& operator=(const complex& rhs) {
    if (this != &rhs) {   // guard against self-assignment (important for deep copy)
        x = rhs.x; y = rhs.y;
    }
    return *this;
}
```

# Friend functions and Classes in C++

Not supported in Java

A **friend** function/class is granted access to the `private` and `protected` members of a class, even though it is **not** a member of that class. You declare the friendship *inside* the class, using the `friend` keyword.

**Friend function example:**
```cpp
class complex {
    int x, y;
public:
    complex(int a=0, int b=0): x(a), y(b) {}
    // grant this non-member function access to private x, y
    friend complex operator+(const complex& a, const complex& b);
    friend ostream& operator<<(ostream& os, const complex& c);
};

// NOT a member (no complex:: prefix), but can touch a.x, b.y directly
complex operator+(const complex& a, const complex& b) {
    return complex(a.x + b.x, a.y + b.y);
}
```
This is the classic use-case: global `operator+` / `operator<<` that need access to private data.

**Friend class example:**
```cpp
class A {
    int secret = 42;
    friend class B;      // B can access A's private members
};
class B {
public:
    void show(A& a) { cout << a.secret; }   // allowed
};
```

**Key properties (common interview points):**
- Friendship is **not mutual**: A declaring B a friend does NOT make A a friend of B. (This is what the note below means.)
- Friendship is **not inherited**: a friend of the base class is not automatically a friend of a derived class.
- Friendship is **not transitive**: friend of a friend is not a friend.
- It is granted *by* the class (declared inside it) — an outside function can't declare *itself* a friend. So it doesn't actually break encapsulation; the class owner chooses who to trust.

Class A is friend of B != Class B is a friend of A







































































































